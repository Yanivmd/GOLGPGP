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

inline void fillMargin(short* field, int sizeX, int sizeY,int val)
{
	for (int r = 0; r < sizeY; r++)
	{
		field[r * sizeX] = val;
		field[r * sizeX + sizeX-1] = val;
	}
	std::fill_n(&field[0],sizeX,val);
	std::fill_n(&field[sizeX*(sizeY-1)],sizeX,val);
}

byte* host(byte* input, int iterations)
{
	if (iterations == 0) {
		return input;
	}

	// globals on CPU
	byte *h_in=NULL, *h_out=NULL;
	short *h_blockGenerations=NULL;
	byte *h_bordersArray = NULL;
	byte *h_bordersArray2 = NULL;

	// Globals on GPU
	byte *d_in=NULL, *d_out=NULL;
	short *d_blockGenerations=NULL;
	byte *d_bordersArray = NULL;
	byte *d_bordersArray2 = NULL;

	// allocated memory
	byte * d_mem = NULL;
	byte * h_mem = NULL;

	const int numberOfVirtualBlockY = (NUMBER_OF_ROWS+NUM_THREADS_Y-1)/NUM_THREADS_Y;
	const int numberOfVirtualBlockX = (NUMBER_OF_COLS+NUM_THREADS_X-1)/NUM_THREADS_X;

	// memory block sizes
	int numOfBlocks = (numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE);
	int sizeOfBordersAry = (numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE)*(NUM_THREADS_X*2+NUM_THREADS_Y*2);
	int field_size = (NUMBER_OF_COLS+GLOBAL_MARGIN_SIZE)*(NUMBER_OF_ROWS+GLOBAL_MARGIN_SIZE);

	int globalMemSize = (numOfBlocks*sizeof(short)) + (sizeOfBordersAry*sizeof(byte)) + (sizeOfBordersAry*sizeof(byte)) + (field_size*sizeof(byte));
	checkCudaErrors(cudaMalloc((void**)&d_mem,globalMemSize + (field_size*sizeof(byte)))); //tried cudaMallocHost, but was not productive
	h_mem = new byte[globalMemSize];

	h_blockGenerations = (short*)h_mem;
	h_bordersArray = ((byte*)h_blockGenerations) + (numOfBlocks*sizeof(short));
	h_bordersArray2 = h_bordersArray + (sizeOfBordersAry*sizeof(byte));
	h_in = h_bordersArray2 + (sizeOfBordersAry*sizeof(byte));

	d_blockGenerations = (short*)d_mem;
	d_bordersArray = ((byte*)d_blockGenerations) + (numOfBlocks*sizeof(short));
	d_bordersArray2 = d_bordersArray + (sizeOfBordersAry*sizeof(byte));
	d_in = d_bordersArray2 + (sizeOfBordersAry*sizeof(byte));
	d_out = d_in + (field_size*sizeof(byte));

	std::fill_n(h_blockGenerations,numOfBlocks,1);
	fillMargin(h_blockGenerations,numberOfVirtualBlockX+GEN_MARGIN_SIZE,numberOfVirtualBlockY+GEN_MARGIN_SIZE,iterations);

	std::fill_n(h_bordersArray,sizeOfBordersAry,0);
	std::fill_n(h_bordersArray2,sizeOfBordersAry,0);

	memcpy(h_in,input,field_size*sizeof(byte));

	dim3 threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3 grid(NUM_BLOCKS_X,1);

	h_out = new byte[field_size*sizeof(byte)];

#ifdef MEASUREMENTS
	cudaEvent_t start,stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	int iters=10;
	for (int i=0; i<iters; i++) {
#endif
		checkCudaErrors(cudaMemcpyAsync(d_mem, h_mem, globalMemSize, cudaMemcpyHostToDevice));

		kernel<numberOfVirtualBlockY,numberOfVirtualBlockX,NUMBER_OF_ROWS,NUMBER_OF_COLS><<<grid,threads>>>(d_in,d_out,iterations,d_bordersArray,d_bordersArray2,d_blockGenerations);

		checkCudaErrors(cudaMemcpyAsync(h_out, d_out, field_size, cudaMemcpyDeviceToHost));
#ifdef MEASUREMENTS
	}

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msec = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
	msec /= iters;

	printf("%dx%d field size, %d generations, %d iterations, %f ms\n",NUMBER_OF_ROWS,NUMBER_OF_COLS,iterations,iters,msec);
#endif

	cudaFree(d_mem);
	delete[] h_mem;

	return h_out;
}

const int gridDimx = NUM_BLOCKS_X;

const int blockDimx = NUM_THREADS_X;
const int blockDimy = NUM_THREADS_Y;

const int totalNumberOfVbsY = ((MAX_NUMBER_ROWS+NUM_THREADS_Y-1)/NUM_THREADS_Y); //rows
const int totalNumberOfVbsX = ((MAX_NUMBER_COLS+NUM_THREADS_X-1)/NUM_THREADS_X); //cols

const int totalNumberOfVBs =  totalNumberOfVbsY*totalNumberOfVbsX ;
const int totalVirtaulBlocksPerSM = (totalNumberOfVBs) / gridDimx;

__forceinline__ __device__ void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows, byte numberOfWarpsToUse)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;

	if (ty == WARPS_FOR_PACKING)
	{

		// copy border UP
		getUPBorder(BordersAryPlace,totalCols,totalRows)[tx] = blockWithMargin[totalColsWithMar + 1+tx];

		// copy border LEFT
		// move past margin, then skip n rows...
		getLEFTBorder(BordersAryPlace,totalCols,totalRows)[tx] = blockWithMargin[(tx%usedRowsNoMar+1) * (totalColsWithMar) + 1];
	} else {
		// copy border Down
		getDOWNBorder(BordersAryPlace,totalCols,totalRows)[tx] = blockWithMargin[usedRowsNoMar * (totalColsWithMar) + (tx%usedColsNoMar+1)];

		// copy border Right
		// move past margin, then skip n rows...
		getRIGHTBorder(BordersAryPlace,totalCols,totalRows)[tx] = blockWithMargin[(tx%usedRowsNoMar+1) * (totalColsWithMar) + usedColsNoMar];
	}
}

__forceinline__ __device__ void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
		int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows, byte numberOfWarpsToUse)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;

	if (ty % numberOfWarpsToUse == 0)
	{
		// LEFT UP
		blockWithMargin[0*totalColsWithMar+0] = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[totalCols-1]; // -1 , cuz 0 based. (no margin!!!)

		// UP
		blockWithMargin[0*totalColsWithMar+1+tx] = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[tx];
	}

	if (ty % numberOfWarpsToUse ==1)
	{
		// RIGHT UP
		blockWithMargin[0*totalColsWithMar + (usedColsNoMar+1)] = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[0];

		// LEFT
		blockWithMargin[(1+(tx%30))*totalColsWithMar + 0] = getRIGHTBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy,totalVBCols,totalCols,totalRows),totalCols,totalRows)[tx%30];
	}

	if (ty % numberOfWarpsToUse ==2)
	{
		// RIGHT
		blockWithMargin[(1+(tx%30))*totalColsWithMar + (usedColsNoMar+1)] = getLEFTBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy,totalVBCols,totalCols,totalRows),totalCols,totalRows)[tx%30];

		// DOWN LEFT
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + 0] = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[totalCols-1]; // -1 cuz 0 based  . (no margin!!!)
	}

	if (ty % numberOfWarpsToUse ==3)
	{
		// DOWN
		blockWithMargin[(usedRowsNoMar +1)*totalColsWithMar+1+tx] = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[tx];

		// DOWN RIGHT
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + (usedColsNoMar+1)] = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[0];
	}
}

__forceinline__ __device__  void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int roundedTotalCols = (numTotalCols+7)/8;
	int col = tx%roundedTotalCols;
	const int row = ty*8 + (tx/roundedTotalCols);
	const int outIndex = row*roundedTotalCols+col;
	const int inIndexMargin = (row+1)*(numTotalCols+MARGIN_SIZE_COLS) + col*8 + 1;
	if ((row < numUsedRows) && (col < numUsedCols)) {
		byte n1 = 0;
		for (int i=0; i<8 && (col < numUsedCols); i++) {
			n1 |= in[inIndexMargin+i] << (i%8);
			col++;
		}
		out[outIndex] = n1;
	}
}

__forceinline__ __device__  void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int roundedTotalCols = (numTotalCols+7)/8;
	const int inIndex = ty*roundedTotalCols+tx/8;
	const int outIndexMargin = (ty+1)*(numTotalCols+MARGIN_SIZE_COLS) + tx + 1;
	if ((tx < numUsedCols) && (ty < numUsedRows)) {
		byte n1 = (in[inIndex] >> (tx%8)) & 0x1;
		out[outIndexMargin] = n1;
	}
} 

__forceinline__ __device__ void check(int numberOfVirtualBlockX,int numberOfVirtualBlockY, int absGenLocInArray,short * blockGenerations,int k)
{
	__threadfence_system();
#pragma unroll
	for (int i=-1; i<=1; i++) {
#pragma unroll
		for (int j=-1; j<=1; j++) {
			while (blockGenerations[(i * numberOfVirtualBlockX+GEN_MARGIN_SIZE) + j + absGenLocInArray] < k)
				__threadfence_system();
		}
	}
}

__forceinline__ __device__ void  eval(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int totalCols, int totalRows)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// i assume the check done to see if we can cals 
	const int numberOfColsWithMar = totalCols + MARGIN_SIZE_COLS;
	byte *ptr = &(srcBlockWithMargin[((ty+1) * numberOfColsWithMar) + (tx+1)]);
	byte *out = &(tarBlockWithMargin[((ty+1) * numberOfColsWithMar) + (tx+1)]);
	byte neighbors = 0;

	neighbors += ptr[-1 * numberOfColsWithMar + -1];
	neighbors += ptr[-1 * numberOfColsWithMar +  0];
	neighbors += ptr[-1 * numberOfColsWithMar +  1];
	neighbors += ptr[ 0 * numberOfColsWithMar + -1];
	neighbors += ptr[ 0 * numberOfColsWithMar +  1];
	neighbors += ptr[ 1 * numberOfColsWithMar + -1];
	neighbors += ptr[ 1 * numberOfColsWithMar +  0];
	neighbors += ptr[ 1 * numberOfColsWithMar +  1];

	if (neighbors == 3 ||
			(ptr[0] == ALIVE && neighbors == 2) ) {
		*out = ALIVE;
	}
	else {
		*out = DEAD;
	}	
}

template<int numberOfVirtualBlockY, int numberOfVirtualBlockX,const int numberOfRows,const int numberOfCols>
__global__ void kernel(byte* input, byte* output, short iterations, byte *bordersArray, byte *bordersArray2, short * blockGenerations)
{
	const int memoryPerVirtualBlock = (blockDimx+MARGIN_SIZE_COLS)*(blockDimy+MARGIN_SIZE_ROWS);

	__shared__ byte work__shared__[memoryPerVirtualBlock];
	__shared__ byte work2__shared__[memoryPerVirtualBlock];

	const int sizeOfPackedVB = ((blockDimx+7)/8)*blockDimy; 
	const int numberOfVBsPerBlock = ((numberOfVirtualBlockY*numberOfVirtualBlockX)+NUM_BLOCKS_X-1)/NUM_BLOCKS_X;

	const int myFirstVB = numberOfVBsPerBlock * blockIdx.x;

	__shared__ byte packed__shared__[sizeOfPackedVB*totalVirtaulBlocksPerSM];

	byte* bordersIn = bordersArray;
	byte* bordersOut = bordersArray2;

	byte *currentWork;
	byte *nextWork;

	currentWork = work__shared__;
	nextWork =  work2__shared__;

	byte* in = input;
	byte* out = output;

	{
		// DOR 0 - read from global
		int virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
		int virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

		for (int packedIndex=0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

			int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
			int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

			int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if ((threadIdx.y < usedRows) && (threadIdx.x < usedCols)) {
				const int numberOfColsWithMar = numberOfCols+GLOBAL_MARGIN_SIZE;
				byte *ptr = &(in[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1]);
				byte *out = &(nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1]);
				byte neighbors = 0;

				neighbors += ptr[-1 * numberOfColsWithMar + -1];
				neighbors += ptr[-1 * numberOfColsWithMar +  0];
				neighbors += ptr[-1 * numberOfColsWithMar +  1];
				neighbors += ptr[ 0 * numberOfColsWithMar + -1];
				neighbors += ptr[ 0 * numberOfColsWithMar +  1];
				neighbors += ptr[ 1 * numberOfColsWithMar + -1];
				neighbors += ptr[ 1 * numberOfColsWithMar +  0];
				neighbors += ptr[ 1 * numberOfColsWithMar +  1];

				if (neighbors == 3 ||
						(ptr[0] == ALIVE && neighbors == 2) ) {
					*out = ALIVE;
				}
				else {
					*out = DEAD;
				}
			}

			__syncthreads();

			if (iterations==1) {
				if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1] = nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1];
				}
			} else {
				if (threadIdx.y < usedRows) {
					if ((threadIdx.y < usedRows) && (threadIdx.y < WARPS_FOR_PACKING))
						packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_PUSH))) {
					share2glob(nextWork,getBordersVBfromXY(bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_PUSH);
				}
			}

			__syncthreads();

			virtualGlobalBlockX++;
			if (virtualGlobalBlockX >= numberOfVirtualBlockX) {
				virtualGlobalBlockX = 0;
				virtualGlobalBlockY++;
			}

			byte* tmp = nextWork;
			nextWork=currentWork;
			currentWork=tmp;
		}

		if (iterations == 1) {
			return;
		}
	}

	{
		// this was once for k= iterations...
		for (int k=1; k<iterations-1; k++)
		{

			int virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
			int virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

			for (int packedIndex=0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

				int absGenLocInArray = ((virtualGlobalBlockY+1) * (numberOfVirtualBlockX+GEN_MARGIN_SIZE)) + virtualGlobalBlockX+1;
				int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

				if (((WARPS_FOR_PACKING + WARPS_FOR_PUSH) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_PUSH + WARPS_FOR_BORDERS))) {
					check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray,blockGenerations,k);
					fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
				}

				unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

				__syncthreads();

				if ((threadIdx.y < usedRows) && (threadIdx.x < usedCols)) {
					eval(currentWork,nextWork,NUM_THREADS_X,NUM_THREADS_Y);
				}

				__syncthreads();

				if ((threadIdx.y < usedRows) && (threadIdx.y < WARPS_FOR_PACKING)) {
					packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_PUSH))) {
					share2glob(nextWork,getBordersVBfromXY(bordersOut,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_PUSH);
				}

				__syncthreads();

				// "NOTIFY"
				blockGenerations[absGenLocInArray] = k+1;

				virtualGlobalBlockX++;
				if (virtualGlobalBlockX >= numberOfVirtualBlockX) {
					virtualGlobalBlockX = 0;
					virtualGlobalBlockY++;
				}

				byte* tmp = nextWork;
				nextWork=currentWork;
				currentWork=tmp;

			}

			byte* tmp = bordersIn;
			bordersIn = bordersOut;
			bordersOut = tmp;
		}
	}

	{
		// DOR K - write to global
		int virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
		int virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

		for (int packedIndex=0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

			int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
			int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

			int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if (((WARPS_FOR_PACKING + WARPS_FOR_BORDERS) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS + WARPS_FOR_BORDERS))) {
				int absGenLocInArray = ((virtualGlobalBlockY+1) * (numberOfVirtualBlockX+GEN_MARGIN_SIZE)) + virtualGlobalBlockX+1;
				check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray,blockGenerations,iterations-1);
				fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
			}

			unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

			__syncthreads();

			if ((threadIdx.y < usedRows) && (threadIdx.x < usedCols)) {
				const int numberOfColsWithMar = NUM_THREADS_X+MARGIN_SIZE_COLS;
				byte *ptr = &(currentWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1]);
				byte *outPtr = &(out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1]);
				byte neighbors = 0;

				neighbors += ptr[-1 * numberOfColsWithMar + -1];
				neighbors += ptr[-1 * numberOfColsWithMar +  0];
				neighbors += ptr[-1 * numberOfColsWithMar +  1];
				neighbors += ptr[ 0 * numberOfColsWithMar + -1];
				neighbors += ptr[ 0 * numberOfColsWithMar +  1];
				neighbors += ptr[ 1 * numberOfColsWithMar + -1];
				neighbors += ptr[ 1 * numberOfColsWithMar +  0];
				neighbors += ptr[ 1 * numberOfColsWithMar +  1];

				if (neighbors == 3 ||
						(ptr[0] == ALIVE && neighbors == 2) ) {
					*outPtr = ALIVE;
				}
				else {
					*outPtr = DEAD;
				}
			}

			virtualGlobalBlockX++;
			if (virtualGlobalBlockX >= numberOfVirtualBlockX) {
				virtualGlobalBlockX = 0;
				virtualGlobalBlockY++;
			}

			byte* tmp = nextWork;
			nextWork=currentWork;
			currentWork=tmp;
		}
	}
}
