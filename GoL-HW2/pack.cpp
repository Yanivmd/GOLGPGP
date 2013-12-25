#include "inc.h"

#ifndef CUDA

// byte packing
void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows,int tx,int ty)
{
	
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

/*__global__*/ void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows, int tx, int ty)
{
	int roundedTotalCols = (numTotalCols+7)/8;
	int inIndex = ty*roundedTotalCols+tx/8;
	int outIndexMargin = (ty+1)*(numTotalCols+MARGIN_SIZE_COLS) + tx + 1;
	if ((tx < numUsedCols) && (ty < numUsedRows)) {
		byte n1 = (in[inIndex] >> (tx%8)) & 0x1;
		out[outIndexMargin] = n1;
	}
} 

#endif