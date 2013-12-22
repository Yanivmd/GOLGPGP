/*
 Omer Katz, 3003457074, omerkatz@cs.technion.ac.il
 Yaniv David, 301226817, yanivd@tx.technion.ac.il
*/

#include <iostream>
#include <fstream>
#include <assert.h>
#include "FieldReader.h"

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

using namespace std;

#define NUM_BLOCKS_X 2
#define NUM_BLOCKS_Y 1
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 30

// make cells on margin of field dead
void clearMargin(byte* field, int sizeX, int sizeY);

// write field to stream (screen or file). Optionally, omit writing the margin.
void writeField(byte* field, int sizeX, int sizeY, int marginSize = 0,
    std::ostream& outs = std::cout);

// simple host implementation of computing the next step
void evolve(byte* orig, byte* next, int sizeX, int sizeY);

// print usage information
void printUsage(int argc, char** argv);

int writeBufToFile(std::string outfilename, std::string postfix, byte* buf, int fieldSizeX, int fieldSizeY)
{
    std::ofstream outf;
	outf.open((outfilename+postfix).c_str());     //< write to file
    if (outf)
    {
        // print to file without a 1-wide margin
        writeField(buf,fieldSizeX+2,fieldSizeY+2,1,outf); 
        outf.close();
    }
	return 0;
}

__global__ void kernel(
		byte* d_in,
		byte* d_out,
		int sizeX,
		int sizeY,
		int iterations,
		int* blockGenerations
		)
{
	int sizeXmargin = sizeX + 2;

	int maxVirtualBlockY = (sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y;
	int maxVirtualBlockX = (sizeX+NUM_THREADS_X-1)/NUM_THREADS_X;

	int virtaulBlockPerSM = (maxVirtualBlockY*maxVirtualBlockX)/gridDim.x;
	int memoryPerVirtualBlock = (blockDim.x+2)*(blockDim.y+2);
	__shared__ byte mem[memoryPerVirtualBlock*virtaulBlockPerSM];

	byte* in = d_in;
	byte* out = d_out;

	for (int k=0; k<iterations; k++) {
		int virtualBlockY = blockIdx.y; + (blockIdx.x / maxVirtualBlockX);
		int virtualBlockX = blockIdx.x % maxVirtualBlockX;
		while (virtualBlockY < maxVirtualBlockY) {
			while (virtualBlockX < maxVirtualBlockX) {
				int absRow = (virtualBlockY * NUM_THREADS_Y) + threadIdx.y;
				int absCol = (virtualBlockX * NUM_THREADS_X) + threadIdx.x;

				int absGenLocInArray = (virtualBlockY * maxVirtualBlockX) + virtualBlockX;
				__threadfence_system();
				for (int i=-1; i<=1; i++) {
					for (int j=-1; j<=1; j++) {
						int genIndex = (i * maxVirtualBlockX) + j + absGenLocInArray;
						if ((genIndex >= 0) && (genIndex < (maxVirtualBlockX * maxVirtualBlockY)))
							while (blockGenerations[genIndex] < k)
								__threadfence_system();
					}
				}

				if ((absRow < sizeY) && (absCol < sizeX)) {
					int absLocInArray = ((absRow+1) * sizeXmargin) + (absCol+1);
				

					byte* ptr = &in[absLocInArray];
					//TODO check neighbors vector
					int neighbors = 0;

					neighbors += ptr[-1 * sizeXmargin + -1];
					neighbors += ptr[-1 * sizeXmargin +  0];
					neighbors += ptr[-1 * sizeXmargin +  1];
					neighbors += ptr[ 0 * sizeXmargin + -1];
					neighbors += ptr[ 0 * sizeXmargin +  1];
					neighbors += ptr[ 1 * sizeXmargin + -1];
					neighbors += ptr[ 1 * sizeXmargin +  0];
					neighbors += ptr[ 1 * sizeXmargin +  1];

					if (neighbors == 3 ||
						(ptr[0] == ALIVE && neighbors == 2) ) {
						out[absLocInArray] = ALIVE;
					}
					else {
						out[absLocInArray] = DEAD;
					}
				}
							
				__syncthreads();
			
				// this is not necessary on last iteration
				blockGenerations[absGenLocInArray] = k+1;

				virtualBlockX += gridDim.x;

			}
			virtualBlockY += virtualBlockX / maxVirtualBlockX;
			virtualBlockX = virtualBlockX % maxVirtualBlockX;
		}

		byte* tmp = in;
		in=out;
		out=tmp;
	}
}

int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
{
	byte *d_in=NULL, *d_out=NULL;
	int *d_generations=NULL;

	int field_size = (sizeX+2)*(sizeY+2);

	cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	int numBlocks = ((sizeX+NUM_THREADS_X-1)/NUM_THREADS_X) * ((sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	checkCudaErrors(cudaMalloc((void**)&d_in,field_size));
	checkCudaErrors(cudaMalloc((void**)&d_out,field_size));
	checkCudaErrors(cudaMalloc((void**)&d_generations,numBlocks*sizeof(int)));

	cudaMemset(d_out, 0, field_size); //TODO delete
	cudaMemset(d_generations, 0, numBlocks*sizeof(int));

	checkCudaErrors(cudaMemcpy(d_in, input, field_size, cudaMemcpyHostToDevice));

	cudaEventRecord(start, NULL);

	// Setup execution parameters
	dim3 threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3 grid(NUM_BLOCKS_X,NUM_BLOCKS_Y);
	kernel<<<grid,threads>>>(d_in, d_out, sizeX, sizeY, iterations, d_generations);

    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    // check for errors during kernel launch
    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess)
    {
    	printf("Kernel launch failed: %s",cudaGetErrorString(err));
    	exit(1);
    }

    float msec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));

	if (iterations % 2 == 0) {
		cudaMemcpy(output, d_in, field_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(input, d_out, field_size, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(output, d_out, field_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(input, d_in, field_size, cudaMemcpyDeviceToHost);
	}

	printf("%dx%d field size, %d generation, %f ms\n",sizeX,sizeY,iterations,msec);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return 0;
}

int cpuSim(int iterations, byte* ptr1, byte* ptr2, int fieldSizeX, int fieldSizeY)
{
	for (int i=0; i<iterations; i++)
    {
        evolve(ptr1,ptr2,fieldSizeX+2,fieldSizeY+2);
        //writeField(ptr2,fieldSizeX+2,fieldSizeY+2);
        byte* tmp = ptr1; ptr1 = ptr2; ptr2 = tmp;
    }
	return 0;
}

int main(int argc, char** argv)
{
    //const int NUM_ARGS = 3;
    //const int FIELD_SIZE_X = 1000;
    //const int FIELD_SIZE_Y = 1000;

    // uncomment to read from program arguments
    //if (argc != NUM_ARGS+1)
    //{
    //    printUsage(argc, argv);
    //    return 1;
    //}
    //string infilename(argv[1]);
    //string outfilename(argv[2]);
    //int iterations = atoi(argv[3]);
    
    string infilename("spaceship.lif");
    string outfilename("spaceship.lif.out");
    int iterations = 30;
  
    PatternBlock tblock;
    FieldReader reader;
    int fieldSizeX = 140;       //<- FIELD_SIZE_X
    int fieldSizeY = 70;        //<- FIELD_SIZE_Y

    byte *cpuin = new byte[(fieldSizeY+2)*(fieldSizeX+2)];
    byte *gpuin = new byte[(fieldSizeY+2)*(fieldSizeX+2)];
    byte *cpuout = new byte[(fieldSizeY+2)*(fieldSizeX+2)];
	byte *gpuout = new byte[(fieldSizeY+2)*(fieldSizeX+2)];

    assert(reader.readFile(infilename));
    assert(reader.buildField(cpuin,fieldSizeX+2,fieldSizeY+2)); //< leave dead margin
    clearMargin(cpuin,fieldSizeX+2,fieldSizeY+2); 
    //writeField(in,fieldSizeX+2,fieldSizeY+2,0);
    
	memcpy(gpuin,cpuin,(fieldSizeY+2)*(fieldSizeX+2));
	writeBufToFile(outfilename,"gpuin",gpuin,fieldSizeX,fieldSizeY);
	host(fieldSizeX,fieldSizeY,gpuin,gpuout,iterations,outfilename);
	
	writeBufToFile(outfilename,"cpuin",cpuin,fieldSizeX,fieldSizeY);
    cpuSim(iterations,cpuin,cpuout,fieldSizeX,fieldSizeY);
    //writeField(ptr1,fieldSizeX+2,fieldSizeY+2,0,cout);  //< print to screen

	if (iterations % 2 == 0) {
		byte *tmp = cpuout;
		cpuout = cpuin;
		cpuin = tmp;
	}
	writeBufToFile(outfilename,"gpu",gpuout,fieldSizeX,fieldSizeY);
	writeBufToFile(outfilename,"cpu",cpuout,fieldSizeX,fieldSizeY);

	int errors = 0;
	for(int i=0;i<fieldSizeX;i++) {
		for(int j=0;j<fieldSizeY;j++) {
			if (cpuout[(j+1)*(fieldSizeX+2) + (i+1)] != gpuout[(j+1)*(fieldSizeX+2) + (i+1)]) {
				errors +=1;
				std::cout << "fucked " << i << " " << j << " (CPU=[" << (int)(cpuout[(j+1)*(fieldSizeX+2) + (i+1)]) << "],GPU=[" << (int)(gpuout[(j+1)*(fieldSizeX+2) + (i+1)]) << "])\n";
				if (errors == 5)
					break;
			}
		}
		if (errors == 5)
			break;
	}
	
	if(errors > 0) {
		std::cout << "Rrrors Detected" << "\n";		
	} else {
		std::cout << "all good\n";
	}

    delete[] cpuin;
    delete[] gpuin;
    delete[] cpuout;
	delete[] gpuout;

    return 0;
}

void clearMargin(byte* field, int sizeX, int sizeY)
{
    for (int r = 0; r < sizeY; r++)
    {
        field[r * sizeX] = DEAD;
        field[r * sizeX + sizeX-1] = DEAD;
    }
    std::fill_n(&field[0],sizeX,DEAD);
    std::fill_n(&field[sizeX*(sizeY-1)],sizeX,DEAD);
}

void writeField(byte* field, int sizeX, int sizeY, int marginSize,
    std::ostream& outs)
{
    // The field may contain a margin of dead cells that should not be printed.
    // If marginSize==0 then the entire field is printed.
    int m = marginSize; 
    outs << (sizeX-2*m) << " " << (sizeY-2*m) << std::endl;
    byte* ptr = field;
    ptr += m * sizeX;   //< skip top margin
    for (int r = 0+m; r < sizeY-m; r++)
    {
        ptr += m;       //< skip left margin
        for (int c=0+m; c < sizeX-m; c++)
        {
            if (*ptr == DEAD)
                outs << '.';
            else
                outs << '*';
            ptr++;
        }
        ptr += m;       //< skip right margin
        outs << std::endl;
    }
}

void evolve(byte* orig, byte* next, int sizeX, int sizeY)
{
    std::fill_n(next,sizeX * sizeY, DEAD);
    for (int y = 1; y < sizeY-1; ++y)
    {
        for (int x = 1; x < sizeX-1; ++x)
        {
            byte* ptr = &orig[y * sizeX + x];
            int neighbors = 0;
			/*
            int state;
            if (ptr[0] == ALIVE)
                state = ALIVE;
            else
                state = DEAD;
			*/
            if (ptr[-1 * sizeX + -1] == ALIVE) neighbors++;
            if (ptr[-1 * sizeX +  0] == ALIVE) neighbors++;
            if (ptr[-1 * sizeX +  1] == ALIVE) neighbors++;
            if (ptr[ 0 * sizeX + -1] == ALIVE) neighbors++;
            if (ptr[ 0 * sizeX +  1] == ALIVE) neighbors++;
            if (ptr[ 1 * sizeX + -1] == ALIVE) neighbors++;
            if (ptr[ 1 * sizeX +  0] == ALIVE) neighbors++;
            if (ptr[ 1 * sizeX +  1] == ALIVE) neighbors++;

            if (neighbors == 3 ||
               (ptr[0] == ALIVE && neighbors == 2) )
                next[y * sizeX + x ] = ALIVE;
        }
    }
    clearMargin(next, sizeX, sizeY);
}

void printUsage(int argc, char** argv)
{
    cout << "Usage: " << argv[0] << " <input-file> <output-file> <num-iterations>" << endl;
}
