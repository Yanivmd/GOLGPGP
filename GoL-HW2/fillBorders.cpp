	
#include "inc.h"


void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int tx,int ty)
{

	// ajust to margin 
	VBx +=1;
	VBy +=1;

	byte* borderPtr;
	// LEFT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[0*totalCols+0] = borderPtr[totalCols-MARGIN_SIZE_COLS-1]; // -1 , cuz 0 based. (no margin!!!)

	// UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<totalCols-(MARGIN_SIZE_COLS-2)-1;col+=32)
		{
			blockWithMargin[0*totalCols+col] = borderPtr[col-1];
		}

	// RIGHT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[0*totalCols + totalCols-(MARGIN_SIZE_COLS-2)-1] = borderPtr[0]; 

	// LEFT
		borderPtr = getRIGHTBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int row=1+tx;row<totalRows-(MARGIN_SIZE_ROWS-2)-1 ;row+=32)
		{
			blockWithMargin[row*totalCols + 0] = borderPtr[row-1];
		}

	// RIGHT
		borderPtr = getLEFTBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int row=1+tx;row<totalRows-(MARGIN_SIZE_ROWS-2) -1 ;row+=32)
		{
			blockWithMargin[row*totalCols + (totalCols-(MARGIN_SIZE_COLS-2)-1)] = borderPtr[row-1];
		}

	// DOWN LEFT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[(totalRows -1) * totalCols + 0] = borderPtr[totalRows-MARGIN_SIZE_ROWS-1]; // -1 cuz 0 based  . (no margin!!!)

	// DOWN
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<=totalCols-MARGIN_SIZE_COLS;col+=32)
		{
			blockWithMargin[(totalRows-1)*totalCols+col] = borderPtr[col-1];
		}

	// DOWN RIGHT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		blockWithMargin[(totalRows-1) * totalCols + totalCols-(MARGIN_SIZE_COLS-2)-1] = borderPtr[0]; 
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
		fillBorders(blockWithMargin,fullBordersArry,1,1,totalVBCols,usedColsnoMar,usedRowsnoMar,usedColsnoMar+MARGIN_SIZE_COLS,usedRowsnoMar+MARGIN_SIZE_ROWS,tx,0);

	
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



