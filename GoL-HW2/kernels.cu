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

void fillMargin(int* field, int sizeX, int sizeY,int val)
{
	for (int r = 0; r < sizeY; r++)
	{
		field[r * sizeX] = val;
		field[r * sizeX + sizeX-1] = val;
	}
	std::fill_n(&field[0],sizeX,val);
	std::fill_n(&field[sizeX*(sizeY-1)],sizeX,val);
}

byte* host(int numberOfCols, int numberOfRows, byte* input, int iterations)
{
	if (iterations == 0) {
		return input;
	}

	byte* output = NULL;

	// globals on CPU
	int *blockGenerations;
	byte *bordersArray;

	// Globals on GPU
	byte *d_in=NULL, *d_out=NULL;
	int *d_blockGenerations=NULL;
	byte *d_bordersArray = NULL;
	byte *d_bordersArray2 = NULL;

	const int numberOfVirtualBlockY = (numberOfRows+NUM_THREADS_Y-1)/NUM_THREADS_Y;
	const int numberOfVirtualBlockX = (numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X;

	int numOfBlocks = (numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE);

	checkCudaErrors(cudaMalloc((void**)&d_blockGenerations,numOfBlocks*sizeof(int)));
	blockGenerations = new int[numOfBlocks];
	std::fill_n(blockGenerations,numOfBlocks,1);
	fillMargin(blockGenerations,numberOfVirtualBlockX+GEN_MARGIN_SIZE,numberOfVirtualBlockY+GEN_MARGIN_SIZE,iterations);

	int sizeOfBordersAry = (numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE)*(NUM_THREADS_X*2+NUM_THREADS_Y*2);

	checkCudaErrors(cudaMalloc((void**)&d_bordersArray,sizeOfBordersAry*sizeof(byte)));
	checkCudaErrors(cudaMalloc((void**)&d_bordersArray2,sizeOfBordersAry*sizeof(byte)));
	bordersArray = new byte[sizeOfBordersAry];
	std::fill_n(bordersArray,sizeOfBordersAry,0);

	checkCudaErrors(cudaMemcpy(d_blockGenerations, blockGenerations, numOfBlocks*sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_bordersArray, bordersArray, sizeOfBordersAry*sizeof(byte), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_bordersArray2, bordersArray, sizeOfBordersAry*sizeof(byte), cudaMemcpyHostToDevice));

	// original stuff
	int field_size = (numberOfCols+GLOBAL_MARGIN_SIZE)*(numberOfRows+GLOBAL_MARGIN_SIZE);
	checkCudaErrors(cudaMalloc((void**)&d_in,field_size*sizeof(byte)));
	checkCudaErrors(cudaMalloc((void**)&d_out,field_size*sizeof(byte)));

	cudaMemset(d_out, 0, field_size); //TODO delete
	checkCudaErrors(cudaMemcpy(d_in, input, field_size, cudaMemcpyHostToDevice));

	dim3 threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3 grid(NUM_BLOCKS_X,1);

#ifdef MEASUREMENTS
	cudaEvent_t start,stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	int iters=10;
	for (int i=0; i<iters; i++) {
#endif
		kernel<<<grid,threads>>>(d_in,d_out,numberOfRows,numberOfCols,numberOfVirtualBlockX,numberOfVirtualBlockY,iterations,d_bordersArray,d_bordersArray2,d_blockGenerations);
#ifdef MEASUREMENTS
	}

	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msec = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
	msec /= iters;

	printf("%dx%d field size, %d generation, %f ms\n",numberOfCols,numberOfRows,iterations,msec);
#endif

	output = new byte[field_size*sizeof(byte)];
	cudaMemcpy(output, d_out, field_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_blockGenerations);
	cudaFree(d_bordersArray);

	delete[] bordersArray;
	delete[] blockGenerations;

	return output;
}

const int gridDimx = NUM_BLOCKS_X;

const int blockDimx = NUM_THREADS_X;
const int blockDimy = NUM_THREADS_Y;

const int totalNumberOfVbsY = ((MAX_NUMBER_ROWS+NUM_THREADS_Y-1)/NUM_THREADS_Y);  //rows
const int totalNumberOfVbsX = ((MAX_NUMBER_COLS+NUM_THREADS_X-1)/NUM_THREADS_X);  //cols

const int totalNumberOfVBs =  totalNumberOfVbsY*totalNumberOfVbsX ;
const int totalVirtaulBlocksPerSM = (totalNumberOfVBs) / gridDimx;


// TODO - all y!=0 are waisted...
__forceinline__ __device__ void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows, byte numberOfWarpsToUse)
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;

	byte *row2Fill;
	int writeIndex;

	int dev8 = tx;

	if (ty % numberOfWarpsToUse == (0 % numberOfWarpsToUse))
	{

		// copy border UP
		row2Fill = getUPBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1;row<=1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=32)
			{
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=32;
			}
		}
	}

	if (ty % numberOfWarpsToUse == (1 % numberOfWarpsToUse))
	{
		// copy border Down
		row2Fill = getDOWNBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1+usedRowsNoMar-1;row<=1+usedRowsNoMar-1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=32)
			{	
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=32;
			}
		}
	}

	if (ty % numberOfWarpsToUse == (2 % numberOfWarpsToUse))
	{
		// copy border LEFT
		row2Fill = getLEFTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1 +dev8;row<=usedRowsNoMar;row+=32)
		{	
			for (int col=1;col<=1;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=32;
			}
		}
	}

	if (ty % numberOfWarpsToUse == (3 % numberOfWarpsToUse))
	{
		// copy border Right
		row2Fill = getRIGHTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1 +dev8;row<=usedRowsNoMar;row+=32)
		{	
			for (int col=usedColsNoMar;col<=usedColsNoMar;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalColsWithMar) + col];
				writeIndex +=32;
			}
		}
	}
}


__forceinline__ __device__ void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
		int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows, byte numberOfWarpsToUse)
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// ajust to margin 
	//VBx +=1;
	//VBy +=1;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;
	byte* borderPtr;

	if (ty % numberOfWarpsToUse == (0 % numberOfWarpsToUse))
	{
		// LEFT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[0*totalColsWithMar+0] = borderPtr[totalCols-1]; // -1 , cuz 0 based. (no margin!!!)

		// UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<totalColsWithMar-(MARGIN_SIZE_COLS-2)-1;col+=32)
		{
			blockWithMargin[0*totalColsWithMar+col] = borderPtr[col-1];
		}
	}

	if (ty % numberOfWarpsToUse ==(1 % numberOfWarpsToUse))
	{
		// RIGHT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[0*totalColsWithMar + (usedColsNoMar+1)] = borderPtr[0]; 

		// LEFT
		byte * ptr1 = getBordersVBfromXY(fullBordersArry,VBx-1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getRIGHTBorder(ptr1,totalCols,totalRows);
		for (int row=1+tx;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2)-1 ;row+=32)
		{
			blockWithMargin[row*totalColsWithMar + 0] = borderPtr[row-1];
		}
	}

	if (ty % numberOfWarpsToUse ==(2 % numberOfWarpsToUse))
	{
		// RIGHT
		byte * ptr2 = getBordersVBfromXY(fullBordersArry,VBx+1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getLEFTBorder(ptr2,totalCols,totalRows);
		for (int row=1+tx;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2) -1 ;row+=32)
		{
			blockWithMargin[row*totalColsWithMar + (usedColsNoMar+1)] = borderPtr[row-1];
		}

		// DOWN LEFT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + 0] = borderPtr[totalCols-1]; // -1 cuz 0 based  . (no margin!!!)
	}
	if (ty % numberOfWarpsToUse ==(3 % numberOfWarpsToUse))
	{
		// DOWN
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<=totalColsWithMar-MARGIN_SIZE_COLS;col+=32)
		{
			blockWithMargin[(usedRowsNoMar +1)*totalColsWithMar+col] = borderPtr[col-1];
		}

		// DOWN RIGHT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + (usedColsNoMar+1)] = borderPtr[0]; 
	}
}







__forceinline__ __device__  void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	int roundedTotalCols = (numTotalCols+7)/8;
	int col = tx%roundedTotalCols;
	int row = ty*8 + (tx/roundedTotalCols);
	int outIndex = row*roundedTotalCols+col;
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
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	int roundedTotalCols = (numTotalCols+7)/8;
	int inIndex = ty*roundedTotalCols+tx/8;
	int outIndexMargin = (ty+1)*(numTotalCols+MARGIN_SIZE_COLS) + tx + 1;
	if ((tx < numUsedCols) && (ty < numUsedRows)) {
		byte n1 = (in[inIndex] >> (tx%8)) & 0x1;
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
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// i assume the check done to see if we can cals 
	int numberOfColsWithMar = totalCols + MARGIN_SIZE_COLS;
	byte *ptr = &(srcBlockWithMargin[((ty+1) * numberOfColsWithMar) + (tx+1)]);
	byte *out = &(tarBlockWithMargin[((ty+1) * numberOfColsWithMar) + (tx+1)]);
	//TODO check neighbors vector
	int neighbors = 0;

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

__global__ void kernel(byte* input, byte* output,const int numberOfRows,const int numberOfCols,
		int numberOfVirtualBlockX, int numberOfVirtualBlockY,
		int iterations, byte *bordersArray, byte *bordersArray2, int * blockGenerations)
{
	const int memoryPerVirtualBlock = (blockDimx+MARGIN_SIZE_COLS)*(blockDimy+MARGIN_SIZE_ROWS);

	__shared__ byte work__shared__[memoryPerVirtualBlock];
	__shared__ byte work2__shared__[memoryPerVirtualBlock];

	// TODO - we only really need to zero the margin...
	for (int i=threadIdx.x;i<memoryPerVirtualBlock;i+=blockDim.x)
	{
		work__shared__[i] = 0;
		work2__shared__[i] = 0;	
	}

	const int sizeOfPackedVB = ((blockDimx+7)/8)*blockDimy; 
	__shared__ byte packed__shared__[sizeOfPackedVB*totalVirtaulBlocksPerSM];

	byte* bordersIn = bordersArray;
	byte* bordersOut = bordersArray2;

	byte *currentWork;
	byte *nextWork;

	currentWork = work__shared__;
	nextWork =  work2__shared__;

	byte* in = input; //was d_
	byte* out = output; // was d_

	// DOR 0 - read from global
	int virtualGlobalBlockY = blockIdx.y + (blockIdx.x / numberOfVirtualBlockX);
	int virtualGlobalBlockX = blockIdx.x % numberOfVirtualBlockX;

	int packedIndex = 0;

	while (virtualGlobalBlockY < numberOfVirtualBlockY) {
		while (virtualGlobalBlockX < numberOfVirtualBlockX) {

			int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
			int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

			int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
				int numberOfColsWithMar = numberOfCols+GLOBAL_MARGIN_SIZE;
				byte *ptr = &(in[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1]);
				byte *out = &(nextWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1]);
				int neighbors = 0;

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

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS))) {
					share2glob(nextWork,getBordersVBfromXY(bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
				}
			}

			// TODO - check if necessary
			__syncthreads();

			virtualGlobalBlockX += gridDimx;
			packedIndex +=1;

			byte* tmp = nextWork;
			nextWork=currentWork;
			currentWork=tmp;
		}

		virtualGlobalBlockY += virtualGlobalBlockX / numberOfVirtualBlockX;
		virtualGlobalBlockX = virtualGlobalBlockX % numberOfVirtualBlockX;
	}

	if (iterations == 1) {
		return;
	}

	__syncthreads(); // tODO check if necessary

	// this was once for k= iterations...
	for (int k=1; k<iterations-1; k++)
	{

		int virtualGlobalBlockY = blockIdx.y + (blockIdx.x / numberOfVirtualBlockX);
		int virtualGlobalBlockX = blockIdx.x % numberOfVirtualBlockX;

		int packedIndex = 0;

		while (virtualGlobalBlockY < numberOfVirtualBlockY) {
			while (virtualGlobalBlockX < numberOfVirtualBlockX) {

				int absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

				check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray,blockGenerations,k);

				if (((WARPS_FOR_PACKING + WARPS_FOR_BORDERS) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS + WARPS_FOR_BORDERS))) {
					fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
				}

				unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

				__syncthreads();

				int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
				int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;


				if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					eval(currentWork,nextWork,NUM_THREADS_X,NUM_THREADS_Y);
				}

				__syncthreads();

				if ((threadIdx.y < usedRows) && (threadIdx.y < WARPS_FOR_PACKING)) {
					packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);
				}

				if ((WARPS_FOR_PACKING <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS))) {
					share2glob(nextWork,getBordersVBfromXY(bordersOut,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
							usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
				}

				__syncthreads();

				// this is not necessary on last iteration
				// "NOTIFY"
				blockGenerations[absGenLocInArray] = k+1;

				virtualGlobalBlockX += gridDimx;
				packedIndex +=1;

				byte* tmp = nextWork;
				nextWork=currentWork;
				currentWork=tmp;
			}

			virtualGlobalBlockY += virtualGlobalBlockX / numberOfVirtualBlockX;
			virtualGlobalBlockX = virtualGlobalBlockX % numberOfVirtualBlockX;
		}

		byte* tmp = bordersIn;
		bordersIn = bordersOut;
		bordersOut = tmp;
	}

	// DOR K - write to global
	virtualGlobalBlockY = blockIdx.y + (blockIdx.x / numberOfVirtualBlockX);
	virtualGlobalBlockX = blockIdx.x % numberOfVirtualBlockX;

	packedIndex = 0;

	while (virtualGlobalBlockY < numberOfVirtualBlockY) {
		while (virtualGlobalBlockX < numberOfVirtualBlockX) {

			int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
			int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

			int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdx.y;
			int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdx.x;

			if (((WARPS_FOR_PACKING + WARPS_FOR_BORDERS) <= threadIdx.y) && (threadIdx.y < (WARPS_FOR_PACKING + WARPS_FOR_BORDERS + WARPS_FOR_BORDERS))) {
				fillBorders(currentWork,bordersIn,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X),usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,WARPS_FOR_BORDERS);
			}

			unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y);

			__syncthreads();

			if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
				int numberOfColsWithMar = NUM_THREADS_X+MARGIN_SIZE_COLS;
				byte *ptr = &(currentWork[(threadIdx.y+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdx.x+1]);
				byte *outPtr = &(out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1]);
				int neighbors = 0;

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

			virtualGlobalBlockX += gridDimx;
			packedIndex +=1;

			byte* tmp = nextWork;
			nextWork=currentWork;
			currentWork=tmp;

		}

		virtualGlobalBlockY += virtualGlobalBlockX / numberOfVirtualBlockX;
		virtualGlobalBlockX = virtualGlobalBlockX % numberOfVirtualBlockX;
	}
}
