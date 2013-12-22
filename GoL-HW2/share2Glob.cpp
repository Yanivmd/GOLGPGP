
#include "inc.h"



// only one warp will work on this...make it work hard! , ty=CONST, tx=0..31 (%?)
void share2glob(byte * blockWithMargin,int usedColsNoMar, int usedRowsNoMar, int otalCols,int otalRows,int *BordersAryPlace,int tx, int ty)
{
	int *row2Fill;
	int writeIndex;

	int dev4 = tx / 8;
	int dev8 = tx % 8;

	if (dev4 == 0)
	{
		// copy border UP
		row2Fill = &(BordersAryPlace[usedColsNoMar*0]);
		writeIndex = dev8;
		for (int row=1;row<=1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=8)
			{
				row2Fill[writeIndex] = blockWithMargin[row * (NumberOfCols2Jump) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4 == 1)	{
		// copy border Down
		row2Fill = &(BordersAryPlace[usedColsNoMar*1]);
		writeIndex = dev8;
		for (int row=1+usedColsNoMar-1;row<=1+NumberOfRowsNoMar-1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=8)
			{	
				row2Fill[writeIndex] = blockWithMargin[row * (NumberOfCols2Jump) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4==2) {
		// copy border LEFT
		row2Fill = &(BordersAryPlace[NumberOfColsNoMar*2]);
		writeIndex = dev8;
		for (int row=1 +dev8;row<=NumberOfRowsNoMar;row+=8)
		{	
			for (int col=1;col<=1;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (NumberOfCols2Jump) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4==3)	{
		// copy border Right
		row2Fill = &(BordersAryPlace[NumberOfColsNoMar*2 + NumberOfRowsNoMar]);
		writeIndex = dev8;
		for (int row=1 + +dev8;row<=NumberOfRowsNoMar;row+=8)
		{	
			for (int col=NumberOfColsNoMar;col<=NumberOfColsNoMar;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (NumberOfCols2Jump) + col];
				writeIndex +=8;
			}
		}
	}
}




	


int test1()
{
	byte packedValues__shared__[] = {0, 0, 0, 0, 0 , 0, 
						             0 ,1 ,2 ,3 ,4 , 0, 
						             0 ,5 ,6 ,7 ,8 , 0, 
						             0 ,9 ,10,11,12, 0, 
						             0 ,13,14,15,16, 0, 
						             0 ,0 ,0 ,0 ,0  ,0};
	const int NumColsNoMar= 4;
	const int  NumRowsNoMar= 4;

	const int NUM_VIRT_BLOCKS =1;

	// 4 - the number of borders for each block (virtual blocks with border near outside the board will still be represetned for clearity)
	// The order is UP,DOWN,LEFT,RIGHT!
	int borders__global__[4 * NumColsNoMar * NUM_VIRT_BLOCKS];

	int NumberOfVirtBLock = 0;


	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__,NumColsNoMar,NumRowsNoMar,NumColsNoMar+2,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),tx,0);

	int expectedRes[] = {1,2,3,4,13,14,15,16,1,5,9,13,4,8,12,16};

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}

int test2()
{

	byte packedValues__shared__[] = {0 ,0 ,0, 0 , 0, 99,99, 
						            0 ,1 ,2 ,3 , 0, 99,99, 
						            0 ,5 ,6 ,7 , 0, 99,99, 
						            0 ,9 ,10,11, 0, 99,99, 
						            0 ,13,14,15, 0, 99,99, 
						            0 ,0 ,0 ,0 , 0, 99,99, };
	const int NumRowsNoMar= 4;
	const int NumColsNoMar= 3;
	const int NUM_VIRT_BLOCKS =1;

	int borders__global__[(2 * NumColsNoMar * NUM_VIRT_BLOCKS)+(2 * NumRowsNoMar * NUM_VIRT_BLOCKS)];

	int expectedRes[] = {1,2,3,13,14,15,1,5,9,13,3,7,11,15};

	int numberOfVirtBbock = 0;

	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__,NumColsNoMar,NumRowsNoMar,NumColsNoMar+2+2,&(borders__global__[(2 * NumColsNoMar * numberOfVirtBbock)+(2 * NumRowsNoMar * numberOfVirtBbock)]),tx,0);

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}

int test3()
{
	const int NumRowsNoMar= 140;
	const int NumColsNoMar= 70;

	byte packedValues__shared[(NumRowsNoMar+2)*(NumColsNoMar+2)];

	for (int i=1;i<NumRowsNoMar+2;i++)
		for (int j=1;j<NumColsNoMar+2;j++)
			packedValues__shared[i*(NumColsNoMar+2)+j] =i*j; 


	const int NUM_VIRT_BLOCKS =1;

	int borders__global__[(2 * NumColsNoMar * NUM_VIRT_BLOCKS)+(2 * NumRowsNoMar * NUM_VIRT_BLOCKS)];

	int expectedRes[140+140+70+70];

	int writerIndex = 0;

	for (int row=1;row<(NumRowsNoMar+1);row++)
	{
		for (int j=1;j<(NumColsNoMar+1);j++)
		{
			if (row==1 || row==(NumRowsNoMar))
			{
				expectedRes[writerIndex++] = row*j;
			}
		}
	}

	for (int i=1;i<(NumRowsNoMar+1);i++)
	{
		for (int j=1;j<(NumColsNoMar+1);j++)
		{
			if (j==1)
			{
				expectedRes[writerIndex++] = i*j;
			}
		}
	}

	for (int i=1;i<(NumRowsNoMar+1);i++)
	{
		for (int j=1;j<(NumColsNoMar+1);j++)
		{
			if (j==(NumColsNoMar))
			{
				expectedRes[writerIndex++] = i*j;
			}
		}
	}

	int numberOfVirtBbock = 0;

	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__BIG,NumColsNoMar,NumRowsNoMar,&(borders__global__[(2 * NumColsNoMar * numberOfVirtBbock)+(2 * NumRowsNoMar * numberOfVirtBbock)]),tx,0);

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}


int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
{
	int err;
	if (test1() != 0)
	{
		err = 1;
	}
	if (test2() != 0)
	{
		err = 2;
	}
	if (test3() != 0)
	{
		err = 3;
	}
	return 0;
}



