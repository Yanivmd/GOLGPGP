#include "inc.h"


#ifndef CUDA

#ifdef CUDA
__forceinline__ __device__
#endif
void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int numberOfWarpsToUse,
#ifndef CUDA
int tx,int ty)
#endif
{
}

#endif