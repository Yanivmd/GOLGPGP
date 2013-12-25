	
#include "inc.h"


#ifndef CUDA

void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int numberOfWarpsToUse,int tx,int ty)
{

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


int fillTest1()
{

	const int usedColsnoMar = 3;
	const int usedRowsnoMar = 2;

	const int totalCols = usedColsnoMar+MARGIN_SIZE_COLS;
	const int totalRows = usedRowsnoMar*MARGIN_SIZE_ROWS;

	byte blockWithMargin[(totalCols)*(totalRows)] ={	0,0 ,0 ,0 ,0 ,0,
														0,10,11,12,0 ,0,
														0,13,14,15,0 ,0,
														0,0 ,0 ,0 ,0 ,0,
	};

	byte fullBordersArry[9*(usedColsnoMar*2+usedRowsnoMar*2)];

	int totalVBCols = 1+VB_MARGIN_SIZE;

	int runningNumber = 1;
	for (int vby=0;vby<=2;vby++)
	{
		for (int vbx=0;vbx<=2;vbx++)
		{
			byte *ptr = getBordersVBfromXY(fullBordersArry,vbx,vby,totalVBCols,totalCols,totalRows);
			for (int k=0;k<(usedColsnoMar*2+usedRowsnoMar*2);k++)
			{
				ptr[k] = runningNumber;
			}
			runningNumber++;
		}
	}


	byte expected[(usedColsnoMar+MARGIN_SIZE_COLS)*(usedRowsnoMar*MARGIN_SIZE_ROWS)] ={1,2 ,2 ,2 ,3 ,0,
																					   4,10,11,12,6 ,0,
																					   4,13,14,15,6 ,0,
																					   7,8 ,8 ,8 ,9 ,0,
	};

	for (int tx=0;tx<32;tx++)
		fillBorders(blockWithMargin,fullBordersArry,1,1,totalVBCols,usedColsnoMar,usedRowsnoMar,usedColsnoMar+MARGIN_SIZE_COLS,usedRowsnoMar+MARGIN_SIZE_ROWS,tx,0,1);

	
	int res = memcmp(expected,blockWithMargin,sizeof(expected));
	return res;
}

int fillTest2()
{

	return 0;
}

int fillTest3()
{
	return 0;
}

//
int fillTester(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
{
	int err;
	if (fillTest1() != 0)
	{
		err = 1;
	}
	return 0;
}

#endif



#include "inc.h"

#ifndef CUDA


void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int numberOfWarpsToUse,int tx, int ty)
{

	//totalCols = MARGIN_SIZE_COLS;
	//totalRows = MARGIN_SIZE_ROWS;

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;

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
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,1,tx,0);

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
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,1,tx,0);

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
		share2glob(packedValues__shared__,&(borders__global__[NumberOfVirtBLock *4 * NumColsNoMar]),NumColsNoMar,NumRowsNoMar,NumColsNoMar+MARGIN_SIZE_COLS,NumRowsNoMar+MARGIN_SIZE_ROWS,1,tx,0);

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