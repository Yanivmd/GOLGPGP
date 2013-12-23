#include "inc.h"

// byte packing
/*__global__*/ void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows, int tx, int ty)
{
	int col = tx*8;
	int roundedTotalCols = ((numTotalCols+7)/8)*8;
	int outIndex = ty*roundedTotalCols+col;
	int inIndexMargin = (ty+1)*(numTotalCols+MARGIN_SIZE_COLS) + col + 1;
	byte n1 = 0;
	if (ty < numUsedRows) {
		for (int i=0; i<8 && (col < numUsedCols); i++) {
			n1 |= in[inIndexMargin] << (col%8);
			col++;
			inIndexMargin++;
		}
	}
	out[outIndex/8] = n1;
}
/*__global__*/ void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows, int tx, int ty)
{
	int roundedTotalCols = (numTotalCols+7)/8*8;
	int inIndex = ty*roundedTotalCols+tx;
	int outIndexMargin = (ty+1)*(numTotalCols+MARGIN_SIZE_COLS) + tx + 1;
	if ((tx < numUsedCols) && (ty < numUsedRows)) {
		byte n1 = (in[inIndex/8] >> (tx%8)) & 0x1;
		out[outIndexMargin] = n1;
	}
} 