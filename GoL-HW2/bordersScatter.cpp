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
	const int tx = ThreadIdx.x;
	const int ty = ThreadIdx.y;
	#endif

	const int totalColsWithMar = totalCols+ MARGIN_SIZE_COLS;
	const int totalRowsWithMar = totalRows + MARGIN_SIZE_ROWS;
	byte* borderPtr;

	
	byte * UP = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
	byte * UP = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
	byte * UP = getDOWNBorder(getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
	
	byte * workingVB;


	
	// LEFT UP
	
		// this will be pushed to first in left, and first in down
		workingVB = getBordersVBfromXY(fullBordersArry,VBx-1,VBy-1,totalVBCols,totalCols,totalRows),totalCols,totalRows);
		
		byte * right = getLEFTBorder(workingVB,totalCols,totalRows);
		byte * up = getUPBorder(workingVB,totalCols,totalRows);
		left[0] 


	// UP
		
	

	
	// RIGHT UP
	
	// LEFT
	

	
	// RIGHT
	

	// DOWN LEFT
	

	
	// DOWN
	

	// DOWN RIGHT
	

}
#endif



#endif