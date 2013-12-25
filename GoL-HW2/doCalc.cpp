

#include "inc.h"

#include "windows.h"

// tx - 0,31 ; ty=0,29
void eval(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int totalCols, int totalRows,int tx, int ty)
{
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
							

int Calctester(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
{
	
	return 0;
}



