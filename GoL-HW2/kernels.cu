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

inline void fillMargin(int* field, int sizeX, int sizeY, int val)
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
	int *h_blockGenerations=NULL;
	byte *h_bordersArray = NULL;
	byte *h_bordersArray2 = NULL;

	// Globals on GPU
	byte *d_in=NULL, *d_out=NULL;
	int *d_blockGenerations=NULL;
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

	int globalMemSize = (numOfBlocks*sizeof(int)) + (sizeOfBordersAry*sizeof(byte)) + (sizeOfBordersAry*sizeof(byte)) + (field_size*sizeof(byte));
	checkCudaErrors(cudaMalloc((void**)&d_mem,globalMemSize + (field_size*sizeof(byte)))); //tried cudaMallocHost, but was not productive
	h_mem = new byte[globalMemSize];

 	h_blockGenerations = (int*)h_mem;
	h_bordersArray = ((byte*)h_blockGenerations) + (numOfBlocks*sizeof(int));
	h_bordersArray2 = h_bordersArray + (sizeOfBordersAry*sizeof(byte));
	h_in = h_bordersArray2 + (sizeOfBordersAry*sizeof(byte));

	d_blockGenerations = (int*)d_mem;
	d_bordersArray = ((byte*)d_blockGenerations) + (numOfBlocks*sizeof(int));
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

		kernel<numberOfVirtualBlockY,numberOfVirtualBlockX,NUMBER_OF_COLS,NUMBER_OF_COLS><<<grid,threads>>>(d_in,d_out,iterations,d_bordersArray,d_bordersArray2,d_blockGenerations);

		checkCudaErrors(cudaMemcpyAsync(h_out, d_out, field_size, cudaMemcpyDeviceToHost));
#ifdef MEASUREMENTS
	}

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msec = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
	msec /= iters;

	printf("%dx%d field size, %d generations, %d iterations, %f ms\n",NUMBER_OF_COLS,NUMBER_OF_ROWS,iterations,iters,msec);
#endif

	cudaFree(d_mem);
	delete[] h_mem;

	return h_out;
}

const int totalNumberOfVbsY = ((MAX_NUMBER_ROWS+NUM_THREADS_Y-1)/NUM_THREADS_Y);  //rows
const int totalNumberOfVbsX = ((MAX_NUMBER_COLS+NUM_THREADS_X-1)/NUM_THREADS_X);  //cols

const int totalNumberOfVBs =  totalNumberOfVbsY*totalNumberOfVbsX ;
const int totalVirtaulBlocksPerSM = (totalNumberOfVBs) / NUM_BLOCKS_X;


// TODO - all y!=0 are waisted...
__forceinline__ __device__ void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows)
{

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;

	byte *row2Fill;
	int writeIndex;

	if (threadIdx.y % WARPS_FOR_BORDERS == (0 % WARPS_FOR_BORDERS))
	{

		// copy border UP
		row2Fill = getUPBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = threadIdx.x;
		for (int row=1;row<=1;row++)
		{
			for (int col=1+threadIdx.x;col<=usedColsNoMar;col+=32)
			{
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=NUM_THREADS_X;
			}
		}
	}

	if (threadIdx.y % WARPS_FOR_BORDERS == (1 % WARPS_FOR_BORDERS))
	{
		// copy border Down
		row2Fill = getDOWNBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = threadIdx.x;
		for (int row=1+usedRowsNoMar-1;row<=1+usedRowsNoMar-1;row++)
		{
			for (int col=1+threadIdx.x;col<=usedColsNoMar;col+=32)
			{	
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=NUM_THREADS_X;
			}
		}
	}

	if (threadIdx.y % WARPS_FOR_BORDERS == (2 % WARPS_FOR_BORDERS))
	{
		// copy border LEFT
		row2Fill = getLEFTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = threadIdx.x;
		for (int row=1 +threadIdx.x;row<=usedRowsNoMar;row+=32)
		{	
			for (int col=1;col<=1;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=NUM_THREADS_X;
			}
		}
	}

	if (threadIdx.y % WARPS_FOR_BORDERS == (3 % WARPS_FOR_BORDERS))
	{
		// copy border Right
		row2Fill = getRIGHTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = threadIdx.x;
		for (int row=1 +threadIdx.x;row<=usedRowsNoMar;row+=32)
		{	
			for (int col=usedColsNoMar;col<=usedColsNoMar;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=NUM_THREADS_X;
			}
		}
	}
}


__forceinline__ __device__ void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
		int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows)
{

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;
	byte* borderPtr;

	if (threadIdx.y % WARPS_FOR_BORDERS == (0 % WARPS_FOR_BORDERS))
	{
		// LEFT UP
		blockWithMargin[0*totalColsWithMar+0] = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[totalCols-1]; // -1 , cuz 0 based. (no margin!!!)

		// UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+threadIdx.x;col<totalColsWithMar-(MARGIN_SIZE_COLS-2)-1;col+=32)
		{
			blockWithMargin[0*totalColsWithMar+col] = borderPtr[col-1];
		}
	}

	if (threadIdx.y % WARPS_FOR_BORDERS ==(1 % WARPS_FOR_BORDERS))
	{
		// RIGHT UP
		blockWithMargin[0*totalColsWithMar + (usedColsNoMar+1)] = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[0];

		// LEFT
		byte * ptr1 = getBordersVBfromXY(fullBordersArry,VBx-1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getRIGHTBorder(ptr1,totalCols,totalRows);
		for (int row=1+threadIdx.x;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2)-1 ;row+=32)
		{
			blockWithMargin[row*totalColsWithMar + 0] = borderPtr[row-1];
		}
	}

	if (threadIdx.y % WARPS_FOR_BORDERS ==(2 % WARPS_FOR_BORDERS))
	{
		// RIGHT
		byte * ptr2 = getBordersVBfromXY(fullBordersArry,VBx+1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getLEFTBorder(ptr2,totalCols,totalRows);
		for (int row=1+threadIdx.x;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2) -1 ;row+=32)
		{
			blockWithMargin[row*totalColsWithMar + (usedColsNoMar+1)] = borderPtr[row-1];
		}

		// DOWN LEFT
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + 0] = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[totalCols-1]; // -1 cuz 0 based  . (no margin!!!)
	}
	if (threadIdx.y % WARPS_FOR_BORDERS ==(3 % WARPS_FOR_BORDERS))
	{
		// DOWN
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+threadIdx.x;col<=totalColsWithMar-MARGIN_SIZE_COLS;col+=32)
		{
			blockWithMargin[(usedRowsNoMar +1)*totalColsWithMar+col] = borderPtr[col-1];
		}

		// DOWN RIGHT
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + (usedColsNoMar+1)] = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows)[0];
	}
}

__forceinline__ __device__  void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows)
{
	const int roundedTotalCols = (numTotalCols+7)/8;
	int col = threadIdx.x%roundedTotalCols;
	const int row = threadIdx.y*8 + (threadIdx.x/roundedTotalCols);
	const int outIndex = row*roundedTotalCols+col;
	int inIndexMargin = (row+1)*(numTotalCols+MARGIN_SIZE_COLS) + col*8 + 1;
	if ((row < numUsedRows) && (col < numUsedCols)) {
		byte n1 = 0;
		for (int i=0; i<8 && (col < numUsedCols); i++) {
			n1 |= in[inIndexMargin] << (i%8);
			col++;
			inIndexMargin++;
		}
		out[outIndex] = n1;
	}
}

__forceinline__ __device__  void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows)
{
	const int roundedTotalCols = (numTotalCols+7)/8;
	const int inIndex = threadIdx.y*roundedTotalCols+threadIdx.x/8;
	const int outIndexMargin = (threadIdx.y+1)*(numTotalCols+MARGIN_SIZE_COLS) + threadIdx.x + 1;
	if ((threadIdx.x < numUsedCols) && (threadIdx.y < numUsedRows)) {
		byte n1 = (in[inIndex] >> (threadIdx.x%8)) & 0x1;
		out[outIndexMargin] = n1;
	}
} 

__forceinline__ __device__ void check(int numberOfVirtualBlockX,int numberOfVirtualBlockY, int absGenLocInArray,int * blockGenerations,int k)
{
	__threadfence_system();
#pragma unroll
	for (int i=-1; i<=1; i++) {
#pragma unroll
		for (int j=-1; j<=1; j++) {
			int genIndex = (i * numberOfVirtualBlockX) + j + absGenLocInArray;
			if ((genIndex >= 0) && (genIndex < (numberOfVirtualBlockX * numberOfVirtualBlockY)))
			{
				while (blockGenerations[genIndex] < k)
					__threadfence_system();
			}
		}
	}
}

// tx - 0,31 ; ty=0,29
__forceinline__ __device__ void  eval(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int totalCols, int totalRows)
{
	// i assume the check done to see if we can cals 
	const int numberOfColsWithMar = totalCols + MARGIN_SIZE_COLS;
	byte *ptr = &(srcBlockWithMargin[((threadIdx.y+1) * numberOfColsWithMar) + (threadIdx.x+1)]);
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
		tarBlockWithMargin[((threadIdx.y+1) * numberOfColsWithMar) + (threadIdx.x+1)] = ALIVE;
	}
	else {
		tarBlockWithMargin[((threadIdx.y+1) * numberOfColsWithMar) + (threadIdx.x+1)] = DEAD;
	}	
}

template<int numberOfVirtualBlockY, int numberOfVirtualBlockX, int numberOfRows, int numberOfCols>
__global__ void kernel(byte* in, byte* out, byte iterations, byte *bordersArray, byte *bordersArray2, int * blockGenerations)
{
	const int memoryPerVirtualBlock = (NUM_THREADS_X+MARGIN_SIZE_COLS)*(NUM_THREADS_Y+MARGIN_SIZE_ROWS);

	__shared__ byte work__shared__[memoryPerVirtualBlock];
	__shared__ byte work2__shared__[memoryPerVirtualBlock];

	// TODO - we only really need to zero the margin...
	for (int i=threadIdx.x;i<memoryPerVirtualBlock;i+=blockDim.x)
	{
		work__shared__[i] = 0;
		work2__shared__[i] = 0;	
	}

	const byte sizeOfPackedVB = ((NUM_THREADS_X+7)/8)*NUM_THREADS_Y;
	__shared__ byte packed__shared__[sizeOfPackedVB*totalVirtaulBlocksPerSM];

	byte* bordersIn = bordersArray;
	byte* bordersOut = bordersArray2;

	byte *currentWork;
	byte *nextWork;

	currentWork = work__shared__;
	nextWork =  work2__shared__;

	const byte numberOfVBsPerBlock = ((numberOfVirtualBlockY*numberOfVirtualBlockX)+NUM_BLOCKS_X-1)/NUM_BLOCKS_X;

	const byte myFirstVB = numberOfVBsPerBlock*blockIdx.x;

	{
		// DOR 0 - read from global
		byte virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
		byte virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

		for(byte packedIndex =0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

			short absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			short absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
				const int numberOfColsWithMar = numberOfCols+GLOBAL_MARGIN_SIZE;
				byte *ptr = &(in[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1]);
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
					nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1] = ALIVE;
				}
				else {
					nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1] = DEAD;
				}
			}

			__syncthreads();

			if (iterations==1) {
				if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					// TODO cache addreess results?
					out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1] = nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1];
				}
			} else {
				byte usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				byte usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

				if (threadIdx.y < usedRows) {
					if ((threadIdx.y < usedRows) && (threadIdx.y < WARPS_FOR_PACKING))
						packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS))) {
					share2glob(nextWork,getBordersVBfromXY(bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}
			}

			// TODO - check if necessary
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
	}

	__syncthreads(); // TODO remove

	if (iterations == 1) {
		return;
	}

	__syncthreads(); // tODO check if necessary

	{
		for (short k=1; k<iterations-1; k++)
		{

			byte virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
			byte virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

			for(byte packedIndex =0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

				byte absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				byte usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				byte usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));


				if (((WARPS_FOR_PACKING + WARPS_FOR_BORDERS) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS + WARPS_FOR_BORDERS))) {
					check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray,blockGenerations,k);
					fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

				__syncthreads();

				if ((threadIdx.x < usedCols) && (threadIdx.y < usedRows)) {
					eval(currentWork,nextWork,NUM_THREADS_X,NUM_THREADS_Y);
				}

				__syncthreads();

				if ((threadIdx.y < usedRows) && (threadIdx.y < WARPS_FOR_PACKING)) {
					packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS))) {
					share2glob(nextWork,getBordersVBfromXY(bordersOut,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
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
		byte virtualGlobalBlockY = myFirstVB / numberOfVirtualBlockX;
		byte virtualGlobalBlockX = myFirstVB % numberOfVirtualBlockX;

		for(byte packedIndex =0; (packedIndex<numberOfVBsPerBlock) && (virtualGlobalBlockY < numberOfVirtualBlockY); packedIndex++) {

			byte usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
			byte usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

			short absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			short absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if (((WARPS_FOR_PACKING + WARPS_FOR_BORDERS) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS + WARPS_FOR_BORDERS))) {
				byte absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray,blockGenerations,iterations-1);
				fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
			}

			unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

			__syncthreads();

			if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
				const byte numberOfColsWithMar = NUM_THREADS_X+MARGIN_SIZE_COLS;
				byte *ptr = &(currentWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1]);
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
					out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1] = ALIVE;
				}
				else {
					out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1] = DEAD;
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
