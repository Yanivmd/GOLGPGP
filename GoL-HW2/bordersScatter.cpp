#include "inc.h"

#ifdef SCATTER_BORDERS

#ifndef CUDA

#ifdef CUDA
__forceinline__ __device__
#endif
void fillBorders(byte * blockWithMargin,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int numberOfWarpsToUse,byte *BordersAryPlace,
#ifndef CUDA
int tx,int ty)
#endif
{
	#ifdef CUDA
	__forceinline__ __device__
	const int tx = ThreadIdx.x;
	const int ty = ThreadIdx.y;
	#endif


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
				blockWithMargin[row * (totalColsWithMar) + col] = row2Fill[writeIndex];
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
				blockWithMargin[row * (totalColsWithMar) + col] = row2Fill[writeIndex];
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
				blockWithMargin[row * (totalColsWithMar) + col] = row2Fill[writeIndex];
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
				blockWithMargin[row * (totalColsWithMar) + col] = row2Fill[writeIndex];
				writeIndex +=32;
			}
		}
	}

}


#ifdef CUDA
__forceinline__ __device__
#endif
void share2glob(byte * blockWithMargin,byte *fullBordersArry,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int numberOfWarpsToUse,
												int VBx,int VBy,int totalVBCols,
#ifndef CUDA
int tx,int ty)
#endif
{
	#ifdef CUDA
	__forceinline__ __device__
	const int tx = ThreadIdx.x;
	const int ty = ThreadIdx.y;
	#endif

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;
	byte* borderPtr;

	if (ty % numberOfWarpsToUse == (0 % numberOfWarpsToUse))
	{
	// LEFT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		borderPtr[totalCols-1] = blockWithMargin[0*totalColsWithMar+0]; // -1 , cuz 0 based. (no margin!!!)

	// UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<totalColsWithMar-(MARGIN_SIZE_COLS-2)-1;col+=32)
		{
			borderPtr[col-1] = blockWithMargin[0*totalColsWithMar+col];
		}
	}

	if (ty % numberOfWarpsToUse ==(1 % numberOfWarpsToUse))
	{
	// RIGHT UP
		borderPtr = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		borderPtr[0] = blockWithMargin[0*totalColsWithMar + (usedColsNoMar+1)]; 

	// LEFT
		byte * ptr1 = getBordersVBfromXY(fullBordersArry,VBx-1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getRIGHTBorder(ptr1,totalCols,totalRows);
		for (int row=1+tx;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2)-1 ;row+=32)
		{
			borderPtr[row-1] = blockWithMargin[row*totalColsWithMar + 0];
		}
	}

	if (ty % numberOfWarpsToUse ==(2 % numberOfWarpsToUse))
	{
	// RIGHT
		byte * ptr2 = getBordersVBfromXY(fullBordersArry,VBx+1,VBy,totalVBCols,totalCols,totalRows);
		borderPtr = getLEFTBorder(ptr2,totalCols,totalRows);
		for (int row=1+tx;row<totalRowsWithMar-(MARGIN_SIZE_ROWS-2) -1 ;row+=32)
		{
			borderPtr[row-1] = blockWithMargin[row*totalColsWithMar + (usedColsNoMar+1)];
		}

	// DOWN LEFT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		borderPtr[totalCols-1] = blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + 0]; // -1 cuz 0 based  . (no margin!!!)
	}
	if (ty % numberOfWarpsToUse ==(3 % numberOfWarpsToUse))
	{
	// DOWN
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		for (int col=1+tx;col<=totalColsWithMar-MARGIN_SIZE_COLS;col+=32)
		{
			borderPtr[col-1] = blockWithMargin[(usedRowsNoMar +1)*totalColsWithMar+col];
		}

	// DOWN RIGHT
		borderPtr = getUPBorder(getBordersVBfromXY(fullBordersArry,VBx+1,VBy+1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		borderPtr[0] = blockWithMargin[(usedRowsNoMar +1) * totalColsWithMar + (usedColsNoMar+1)]; 
	}

}
#endif



#endif