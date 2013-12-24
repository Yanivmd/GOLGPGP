
#include "inc.h"

#ifndef CUDA

//const int gridDimx = NUM_BLOCKS_X;
const int gridDimy = NUM_BLOCKS_Y;
const int gridDimx = NUM_BLOCKS_X;

const int blockDimx = NUM_THREADS_X;
const int blockDimy = NUM_THREADS_Y;

const int totalNumberOfVbsY = ((MAX_NUMBER_ROWS+NUM_THREADS_Y-1)/NUM_THREADS_Y);  //rows
const int totalNumberOfVbsX = ((MAX_NUMBER_COLS+NUM_THREADS_X-1)/NUM_THREADS_X);  //cols

const int totalNumberOfVBs =  totalNumberOfVbsY*totalNumberOfVbsX ;
const int totalVirtaulBlocksPerSM = (totalNumberOfVBs) / gridDimx;




/*
void getFirstVB(int * vbx, int * vby, int blockIdxx,int blockIdxy)
{
		int virtualBlockY = blockIdxy; + (blockIdxx / maxVirtualBlockX);
		int virtualBlockX = blockIdxx % maxVirtualBlockX;
}
*/

int *blockGenerations;

byte *bordersArray;


void check(int numberOfVirtualBlockX,int numberOfVirtualBlockY, int absGenLocInArray)
{
	//__threadfence_system();
	for (int i=-1; i<=1; i++) {
		for (int j=-1; j<=1; j++) {
			int genIndex = (i * numberOfVirtualBlockX) + j + absGenLocInArray;
			if ((genIndex >= 0) && (genIndex < (numberOfVirtualBlockX * numberOfVirtualBlockY)))
			{
				/*
				while (blockGenerations[genIndex] < k)
					__threadfence_system();
				*/
			}
						
		}
	}
}



// COPY TO KERNEL!!

const int memoryPerVirtualBlock = (blockDimx+MARGIN_SIZE_COLS)*(blockDimy+MARGIN_SIZE_ROWS);

byte work__shared__[memoryPerVirtualBlock] = {0};
byte work2__shared__[memoryPerVirtualBlock] = {0};
const int sizeOfPackedVB = ((blockDimx+7)/8)*blockDimy; 
byte packed__shared__[sizeOfPackedVB*totalVirtaulBlocksPerSM];

// END OF COPY

void kernel(byte* input, byte* output,const int numberOfRows,const int numberOfCols,
	int numberOfVirtualBlockX, int numberOfVirtualBlockY,
	int iterations)
{
	//int sizeXmargin = GLOBAL_MARGIN_SIZE + numberOfCols;

	const int numUsedVirtualBlocks = ((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X) * ((numberOfRows+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	/// here we need to allocate all the shared

	
	byte *currentWork;
	byte *nextWork;

	currentWork = work__shared__;
	//lastWork =
	nextWork =  work2__shared__;

	byte* in = input; //was d_
	byte* out = output; // was d_

	// DOR 0 - read from globla
	for (int blockIdxx=0;blockIdxx<NUM_BLOCKS_X;blockIdxx++)
		for (int blockIdxy=0;blockIdxy<NUM_BLOCKS_Y;blockIdxy++)
	{
		int virtualGlobalBlockY = blockIdxy + (blockIdxx / numberOfVirtualBlockX);
		int virtualGlobalBlockX = blockIdxx % numberOfVirtualBlockX;

		int packedIndex = 0;
		//byte *ptr2MySharedBlock = currentWork;

		while (virtualGlobalBlockY < numberOfVirtualBlockY) {
			while (virtualGlobalBlockX < numberOfVirtualBlockX) {

				int absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				//check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray);

				int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

				for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
					for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
				{

					int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
					int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;

					if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
						nextWork[(threadIdxy+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdxx+1] = in[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1];
					}

					//fillBorders(currentWork,bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,totalNumberOfVbsX,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

					//unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

					//__syncthreads();
					//if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					//	eval(currentWork,nextWork,usedCols,usedRows,threadIdxx,threadIdxy);
					//}

					//__syncthreads();
				}

				for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
					for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
					{
						int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
						int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;
						

						if (threadIdxy < usedRows) {
							//if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
								packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
							//}

							share2glob(nextWork,getBordersVBfromXY(bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
								usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
						}
			
						// this is not necessary on last iteration
						// "NOTIFY"
						//blockGenerations[absGenLocInArray] = k+1;

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
	//

	// this was once for k= iterations...
	for (int k=0; k<iterations; k++) 
		for (int blockIdxx=0;blockIdxx<NUM_BLOCKS_X;blockIdxx++)
		for (int blockIdxy=0;blockIdxy<NUM_BLOCKS_Y;blockIdxy++)
	{

		int virtualGlobalBlockY = blockIdxy + (blockIdxx / numberOfVirtualBlockX);
		int virtualGlobalBlockX = blockIdxx % numberOfVirtualBlockX;

		int packedIndex = 0;
		//byte *ptr2MySharedBlock = currentWork;

		while (virtualGlobalBlockY < numberOfVirtualBlockY) {
			while (virtualGlobalBlockX < numberOfVirtualBlockX) {

				int absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));
				

				for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
					for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
				{
				
				check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray);

				int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
				int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;

				fillBorders(currentWork,bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,((numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X)+GEN_MARGIN_SIZE,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

				unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
				}

				for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
				for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
				{

					int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
					int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;

				//__syncthreads();
				if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					eval(currentWork,nextWork,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
				}
				}

				//__syncthreads();

				for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
				for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
				{

				int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
				int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;

				if (threadIdxy < usedRows) {
					//if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
						packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
					//}

					share2glob(nextWork,getBordersVBfromXY(bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
						usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
				}
				}
			
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
	}

	// DOR K - write to global
		for (int blockIdxx=0;blockIdxx<NUM_BLOCKS_X;blockIdxx++)
		for (int blockIdxy=0;blockIdxy<NUM_BLOCKS_Y;blockIdxy++)
	{
	    int virtualGlobalBlockY = blockIdxy + (blockIdxx / numberOfVirtualBlockX);
		int virtualGlobalBlockX = blockIdxx % numberOfVirtualBlockX;

		int packedIndex = 0;
		//byte *ptr2MySharedBlock = currentWork;

		while (virtualGlobalBlockY < numberOfVirtualBlockY) {
			while (virtualGlobalBlockX < numberOfVirtualBlockX) {

				int absGenLocInArray = (virtualGlobalBlockY * numberOfVirtualBlockX) + virtualGlobalBlockX;
				//check(numberOfVirtualBlockX,numberOfVirtualBlockY,absGenLocInArray);
				int usedCols = min(NUM_THREADS_X,numberOfCols-(virtualGlobalBlockX * NUM_THREADS_X));
				int usedRows = min(NUM_THREADS_Y,numberOfRows-(virtualGlobalBlockY * NUM_THREADS_Y));

				for (int threadIdxx=0;threadIdxx<NUM_THREADS_X;threadIdxx++)
				for (int threadIdxy=0;threadIdxy<NUM_THREADS_Y;threadIdxy++)
				{

					int absRow = (virtualGlobalBlockY * NUM_THREADS_Y) + threadIdxy;
					int absCol = (virtualGlobalBlockX * NUM_THREADS_X) + threadIdxx;


				
					//fillBorders(currentWork,bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,totalNumberOfVbsX,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

					unpacker(&packed__shared__[packedIndex*sizeOfPackedVB],currentWork,usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

					//__syncthreads();
					//if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
					//	eval(currentWork,nextWork,usedCols,usedRows,threadIdxx,threadIdxy);
					//}

					//__syncthreads();

					//packer(nextWork,&packed__shared__[packedIndex*sizeOfPackedVB],usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);

					//share2glob(nextWork,getBordersVBfromXY(bordersArray,virtualGlobalBlockX,virtualGlobalBlockY,numberOfVirtualBlockX,NUM_THREADS_X,NUM_THREADS_Y),
					//	usedCols,usedRows,NUM_THREADS_X,NUM_THREADS_Y,threadIdxx,threadIdxy);
			
					// this is not necessary on last iteration
					// "NOTIFY"
					//blockGenerations[absGenLocInArray] = k+1;
					if ((absRow < numberOfRows) && (absCol < numberOfCols)) {
						out[(absRow+1)*(numberOfCols+GLOBAL_MARGIN_SIZE)+absCol+1] = currentWork[(threadIdxy+1)*(NUM_THREADS_X+MARGIN_SIZE_COLS)+threadIdxx+1];
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
	//
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

int host(int numberOfCols, int numberOfRows, byte* input, byte* output, int iterations)
{
	
	const int numberOfVirtualBlockY = (numberOfRows+NUM_THREADS_Y-1)/NUM_THREADS_Y;
	const int numberOfVirtualBlockX = (numberOfCols+NUM_THREADS_X-1)/NUM_THREADS_X;

	blockGenerations = new int[(numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE)];
	//memset(blockGenerations,0,(numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE));
	std::fill_n(blockGenerations,(numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE),0);
	fillMargin(blockGenerations,numberOfVirtualBlockX+GEN_MARGIN_SIZE,numberOfVirtualBlockY+GEN_MARGIN_SIZE,iterations);

	bordersArray = new byte[(numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE)*(NUM_THREADS_X*2+NUM_THREADS_Y*2)];
	std::fill_n(bordersArray,(numberOfVirtualBlockY+GEN_MARGIN_SIZE)*(numberOfVirtualBlockX+GEN_MARGIN_SIZE)*(NUM_THREADS_X*2+NUM_THREADS_Y*2),0);

	kernel(input,output,numberOfRows,numberOfCols,numberOfVirtualBlockX,numberOfVirtualBlockY,iterations);
	

	delete[] bordersArray;
	delete[] blockGenerations;

	return 0;
}


int hostTest1()
{

	byte input[] = {0,0,0,0,0,
					0,1,0,1,0,
					0,0,1,0,0,
					0,1,0,1,0,
					0,0,0,0,0,
	};

	byte expected[] = {0,0,0,0,0,
					   0,0,1,0,0,
					   0,1,0,1,0,
					   0,0,1,0,0,
					   0,0,0,0,0,
	};

	byte output[sizeof(input)/sizeof(byte)];
	memset(output,0,sizeof(input)/sizeof(byte));

	host(3,3,input,output,1);
	int res= memcmp(output,expected,sizeof(expected)/sizeof(byte));
	return res;
}

int hostTest2()
{

	byte input[] = {0,0,0,0,0,
					   0,0,1,0,0,
					   0,1,0,1,0,
					   0,0,1,0,0,
					   0,0,0,0,0,
	};

	byte expected[] = {0,0,0,0,0,
					   0,0,1,0,0,
					   0,1,0,1,0,
					   0,0,1,0,0,
					   0,0,0,0,0,
	};

	byte output[sizeof(input)/sizeof(byte)];
	memset(output,0,sizeof(input)/sizeof(byte));

	host(3,3,input,output,1);
	int res= memcmp(output,expected,sizeof(expected)/sizeof(byte));
	return res;
}

int hostTest3()
{

	byte input[] = {0,0,0,0,0,0,
					0,1,0,1,0,0,
					0,0,1,0,0,0,
					0,1,0,1,0,0,
					0,0,0,0,0,0,
	};

	byte expected[] = {0,0,0,0,0,0,
					   0,0,1,0,0,0,
					   0,1,0,1,0,0,
					   0,0,1,0,0,0,
					   0,0,0,0,0,0,
	};

	byte output[sizeof(input)/sizeof(byte)];
	memset(output,0,sizeof(input)/sizeof(byte));

	host(4,3,input,output,1);
	int res= memcmp(output,expected,sizeof(expected)/sizeof(byte));
	return res;
}


int testHost(int numberOfCols, int numberOfRows, byte* input, byte* output, int iterations, string outfilename)
{

	int res;
	res= hostTest1();
	if (res!=0)
	{
		res = 1;
	}
	res = hostTest2();
	if (res!=0)
	{
		res = 2;
	}

	res = hostTest3();
	if (res!=0)
	{
		res = 3;
	}

	
	return res;
}
#endif