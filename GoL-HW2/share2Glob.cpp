
#include "inc.h"

#ifndef CUDA
/*
// only one warp will work on this...make it work hard! , ty=CONST, tx=0..31 (%?)
void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int tx, int ty)
{
	byte *row2Fill;
	int writeIndex;

	int dev4 = tx / 8;
	int dev8 = tx % 8;

	if (dev4 == 0)
	{
		// copy border UP
		row2Fill = getUPBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1;row<=1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=8)
			{
				row2Fill[writeIndex] = blockWithMargin[row * (totalCols) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4 == 1)	{
		// copy border Down
		row2Fill = getDOWNBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1+usedRowsNoMar-1;row<=1+usedRowsNoMar-1;row++)
		{
			for (int col=1+dev8;col<=usedColsNoMar;col+=8)
			{	
				row2Fill[writeIndex] = blockWithMargin[row * (totalCols) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4==2) {
		// copy border LEFT
		row2Fill = getLEFTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1 +dev8;row<=usedRowsNoMar;row+=8)
		{	
			for (int col=1;col<=1;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalCols) + col];
				writeIndex +=8;
			}
		}
	} else if (dev4==3)	{
		// copy border Right
		row2Fill = getRIGHTBorder(BordersAryPlace,totalCols,totalRows);
		writeIndex = dev8;
		for (int row=1 + +dev8;row<=usedRowsNoMar;row+=8)
		{	
			for (int col=usedColsNoMar;col<=usedColsNoMar;col++)
			{
				// move past margin, then skip n rows...
				row2Fill[writeIndex] = blockWithMargin[row * (totalCols) + col];
				writeIndex +=8;
			}
		}
	}
}
*/

void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int tx, int ty)
{

	//totalCols = MARGIN_SIZE_COLS;
	//totalRows = MARGIN_SIZE_ROWS;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;

	byte *row2Fill;
	int writeIndex;

		int dev8 = tx;
		
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


int test1()
{
	byte packedValues__shared__[] = {0, 0, 0, 0, 0 , 0, 0,
						             0 ,1 ,2 ,3 ,4 , 0, 0,
						             0 ,5 ,6 ,7 ,8 , 0, 0,
						             0 ,9 ,10,11,12, 0, 0,
						             0 ,13,14,15,16, 0, 0,
						             0 ,0 ,0 ,0 ,0  ,0, 0, };
	const int NumColsNoMar= 4;
	const int NumRowsNoMar= 4;

	const int NUM_VIRT_BLOCKS =1;

	// 4 - the number of borders for each block (virtual blocks with border near outside the board will still be represetned for clearity)
	// The order is UP,DOWN,LEFT,RIGHT!
	byte borders__global__[4 * NumColsNoMar * NUM_VIRT_BLOCKS];

	int NumberOfVirtBLock = 0;

	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,tx,0);

	byte expectedRes[] = {1,2,3,4,13,14,15,16,1,5,9,13,4,8,12,16};

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}

int test2()
{

	byte packedValues__shared__[] = {0 ,0 ,0 ,0 , 0, 0, 
						             0 ,1 ,2 ,3 , 0, 0, 
						             0 ,5 ,6 ,7 , 0, 0, 
						             0 ,9 ,10,11, 0, 0, 
						             0 ,13,14,15, 0, 0, 
						             0 ,0 ,0 ,0 , 0, 0, };
	const int NumRowsNoMar= 4;
	const int NumColsNoMar= 3;
	const int NUM_VIRT_BLOCKS =1;

	byte borders__global__[(2 * NumColsNoMar * NUM_VIRT_BLOCKS)+(2 * NumRowsNoMar * NUM_VIRT_BLOCKS)];

	byte expectedRes[] = {1,2,3,13,14,15,1,5,9,13,3,7,11,15};

	int NumberOfVirtBLock = 0;

	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,tx,0);

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}

int test3()
{
	const int NumRowsNoMar= 140;
	const int NumColsNoMar= 70;

	byte packedValues__shared__[(NumRowsNoMar+MARGIN_SIZE_ROWS)*(NumColsNoMar+MARGIN_SIZE_COLS)];

	for (int i=1;i<NumRowsNoMar+MARGIN_SIZE_ROWS;i++)
		for (int j=1;j<NumColsNoMar+MARGIN_SIZE_COLS;j++)
			packedValues__shared__[i*(NumColsNoMar+MARGIN_SIZE_COLS)+j] =i*j; 


	const int NUM_VIRT_BLOCKS =1;

	byte borders__global__[(2 * NumColsNoMar * NUM_VIRT_BLOCKS)+(2 * NumRowsNoMar * NUM_VIRT_BLOCKS)];

	byte expectedRes[140+140+70+70];

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

	int NumberOfVirtBLock = 0;

	for (int tx=0;tx<32;tx++)
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,tx,0);

	return memcmp(expectedRes,borders__global__,sizeof(expectedRes));
}

//
int shareTester(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
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



#endif