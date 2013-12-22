

#include "inc.h"

#include "windows.h"

// tx - 0,31 ; ty=0,29
void doIterationOnShared(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int NumberOfColsNoMar, int NumberOfRowsNoMar,int tx, int ty)
{
	// i assume the check done to see if we can cals 

	int numberOfColsWithMar = NumberOfColsNoMar + 2;
	byte *ptr = &(srcBlockWithMargin[((tx+1) * NumberOfColsNoMar) + (ty+1)]);
	byte *out = &(tarBlockWithMargin[((tx+1) * NumberOfColsNoMar) + (ty+1)]);
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
		out = ALIVE;
	}
	else {
		out = DEAD;
	}	
}
							

int tester(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename)
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



