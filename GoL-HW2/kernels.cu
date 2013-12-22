
#include "inc.h"


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

/*
__global__ void packer(byte *d_in, int *d_packedIn, int sizeX, int sizeY)
{
	__shared__ int packedValues[NUM_THREADS_X*NUM_THREADS_Y/32];

	int myInIndex = (blockIdx.y*blockDim.y+threadIdx.y+1)*(sizeX+2) + (blockIdx.x*blockDim.x+threadIdx.x+1);
	int myOutIndex = ((blockIdx.y*blockDim.y+threadIdx.y)*(sizeX) + (blockIdx.x*blockDim.x+threadIdx.x)) / 32;
	if (myInIndex < ((sizeX+2)*(sizeY+2)))
	{
		int mySharedIndex = ((threadIdx.y * NUM_THREADS_X) + threadIdx.x)/32;
		packedValues[mySharedIndex] =0;
		int n1 = (int)(d_in[myInIndex]);
		n1 = n1 << (myInIndex % 32);

		atomicOr(&(packedValues[mySharedIndex]),n1);
		__syncthreads();

		d_packedIn[myOutIndex] = packedValues[mySharedIndex];
	}
}
*/
/*
//int packing
__global__ void packer(byte *d_in, int *d_packedIn, int sizeX, int sizeY)
{
	int myOutIndex = ((blockIdx.y*blockDim.y+threadIdx.y)*(sizeX) + (blockIdx.x*blockDim.x+threadIdx.x)) / 32;
	int myinIndex = (blockIdx.y*blockDim.y+threadIdx.y+1)*(sizeX+2) + (blockIdx.x*blockDim.x+threadIdx.x+1) ;
	if (myinIndex < ((sizeX+2)*(sizeY+2)))
	{
		int n1 = (((int)d_in[myinIndex]) << (myinIndex % 32)); 
		atomicOr(&(d_packedIn[myOutIndex]),n1);
	}
}
__global__ void unpacker(byte *d_out, int *d_packedOut, int sizeX, int sizeY)
{
	int myinIndex = ((blockIdx.y*blockDim.y+threadIdx.y)*(sizeX) + (blockIdx.x*blockDim.x+threadIdx.x)) / 32;
	int myOutIndex = (blockIdx.y*blockDim.y+threadIdx.y+1)*(sizeX+2) + (blockIdx.x*blockDim.x+threadIdx.x+1) ;
	if (myOutIndex < ((sizeX+2)*(sizeY+2)))
	{
		byte n1 = (d_packedOut[myinIndex] >> (myOutIndex % 32)) & 0x1; 
		d_out[myOutIndex] = n1;
	}
}
*/

// byte packing
/*__global__*/ void packer(byte *d_in, byte *d_packedIn, int sizeX, int sizeY, int bx, int by, int tx, int ty)
{
	//int row = blockIdx.y*blockDim.y+threadIdx.y;
	//int col = blockIdx.x*blockDim.x+threadIdx.x;
	int row = by*((sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y)+ty;
	int col = bx*((sizeX+NUM_THREADS_X-1)/NUM_THREADS_X)+tx;
	int outIndex = (row*sizeX+col);
	if (outIndex < (sizeX*sizeY)) {
		byte n1 = 0;
		for (int i=0; i<8; i++) {
			int inIndexMargin = (row+1)*(sizeX+2) + col + 1;
			n1 += ((d_in[inIndexMargin]) << (outIndex%8));
			col++;
			if (col>=sizeX) {
				row++;
				col -= sizeX;
			}
		}
		d_packedIn[outIndex/8] = n1;
	}
}
/*__global__*/ void unpacker(byte *d_out, byte *d_packedOut, int sizeX, int sizeY, int bx, int by, int tx, int ty)
{
	//int row = blockIdx.y*blockDim.y+threadIdx.y;
	//int col = blockIdx.x*blockDim.x+threadIdx.x;
	int row = by*((sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y)+ty;
	int col = bx*((sizeX+NUM_THREADS_X-1)/NUM_THREADS_X)+tx;
	int outIndexMargin = (row+1)*(sizeX+2) + col + 1;
	int inIndex = (row*sizeX+col);
	if (inIndex < (sizeX*sizeY)) {
		byte n1 = (d_packedOut[inIndex/8] >> (inIndex%8)) & 0x1;
		d_out[outIndexMargin]=n1;
	}
}


int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
{
	byte *d_in=NULL, *d_out=NULL;
	int *d_packedIn=NULL, *d_packedOut=NULL;
	int *d_generations=NULL;

	int field_size = (sizeX+2)*(sizeY+2);

	int numBlocks = ((sizeX+NUM_THREADS_X-1)/NUM_THREADS_X) * ((sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	checkCudaErrors(cudaMalloc((void**)&d_in,field_size*sizeof(byte)));
	checkCudaErrors(cudaMalloc((void**)&d_out,field_size*sizeof(byte)));
	checkCudaErrors(cudaMalloc((void**)&d_packedIn,(field_size/8)));
	checkCudaErrors(cudaMalloc((void**)&d_packedOut,(field_size/8)));
	checkCudaErrors(cudaMalloc((void**)&d_generations,numBlocks*sizeof(int)));

	cudaMemset(d_out, 0, field_size); //TODO delete
	cudaMemset(d_packedIn, 0, (field_size/8)); 
	cudaMemset(d_generations, 0, numBlocks*sizeof(int));

	checkCudaErrors(cudaMemcpy(d_in, input, field_size, cudaMemcpyHostToDevice));

	cudaError_t err;

	byte* packed = new byte[sizeX*sizeY];
	for (int i=0; i<(sizeX+NUM_THREADS_X-1)/NUM_THREADS_X; i++)
		for (int j=0; j<(sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y; j++)
			for (int l=0; l<NUM_THREADS_X; l++)
				for (int k=0; k<NUM_THREADS_Y; k++)
					packer(input,packed,sizeX,sizeY,i,j,l,k);

	for (int i=0; i<(sizeX+NUM_THREADS_X-1)/NUM_THREADS_X; i++)
		for (int j=0; j<(sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y; j++)
			for (int l=0; l<NUM_THREADS_X; l++)
				for (int k=0; k<NUM_THREADS_Y; k++)
					unpacker(output,packed,sizeX,sizeY,i,j,l,k);
	int res  = memcmp(input,output,field_size);
	free(packed);
	/*
	dim3 packerThreads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3 packerGrid((sizeX+NUM_THREADS_X-1)/NUM_THREADS_X,(sizeY+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	packer<<<packerGrid,packerThreads>>>(d_in, d_packedIn, sizeX, sizeY);

	if ((err = cudaGetLastError()) != cudaSuccess)
    {
    	printf("packer launch failed: %s",cudaGetErrorString(err));
    	exit(1);
    }

	cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	// Setup execution parameters
	dim3 threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3 grid(NUM_BLOCKS_X,NUM_BLOCKS_Y);
	//kernel<<<grid,threads>>>(d_in, d_out, sizeX, sizeY, iterations, d_generations);

	if ((err = cudaGetLastError()) != cudaSuccess)
    {
    	printf("Kernel launch failed: %s",cudaGetErrorString(err));
    	exit(1);
    }

	checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

	//TODO   d_packedOut
	unpacker<<<packerGrid,packerThreads>>>(d_out, d_packedIn, sizeX, sizeY);

    // check for errors during kernel launch
   
    if ((err = cudaGetLastError()) != cudaSuccess)
    {
    	printf("unpacker launch failed: %s",cudaGetErrorString(err));
    	exit(1);
    }
	
    float msec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
	

	//TODO!!!!! change back to 0
	if (iterations % 2 == 1) {
		cudaMemcpy(output, d_in, field_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(input, d_out, field_size, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(output, d_out, field_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(input, d_in, field_size, cudaMemcpyDeviceToHost);
	}


//	int res  = memcmp(input,output,field_size);

//	printf("%dx%d field size, %d generation, %f ms\n",sizeX,sizeY,iterations,msec);

    cudaFree(d_in);
    cudaFree(d_out);
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
	*/
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
//	__shared__ byte mem[memoryPerVirtualBlock*virtaulBlockPerSM];

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