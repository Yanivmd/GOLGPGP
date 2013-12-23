#include "FieldReader.h"

// CUDA runtime

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>



#define NUM_BLOCKS_X 2
#define NUM_BLOCKS_Y 1
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 30

#define MARGIN_SIZE_COLS 2

using namespace std;

__global__ void kernel(
		byte* d_in,
		byte* d_out,
		int sizeX,
		int sizeY,
		int iterations,
		int* blockGenerations
		);

int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename);
#else

#include "windows.h"

using namespace std;

#define NUM_BLOCKS_X 1
#define NUM_BLOCKS_Y 1
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 30

#define MARGIN_SIZE_COLS 3
#define MARGIN_SIZE_ROWS 2

#define GLOBAL_MARGIN_SIZE 2

#define MAX_NUMBER_COLS 1000
#define MAX_NUMBER_ROWS 1000


#define VB_MARGIN_SIZE 2

#define GEN_MARGIN_SIZE 2

void kernel(
		byte* d_in,
		byte* d_out,
		int sizeX,
		int sizeY,
		int iterations,
		int* blockGenerations
		);

int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename);

inline byte * getUPBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols-MARGIN_SIZE_COLS)*0]);
}

inline byte * getDOWNBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols-MARGIN_SIZE_COLS)*1]);
}

inline byte * getLEFTBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols-MARGIN_SIZE_COLS)*2]);
}

inline byte * getRIGHTBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols-MARGIN_SIZE_COLS)*2 + (totalRows-MARGIN_SIZE_ROWS)*1]);
}


void fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int tx,int ty);

void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows, int tx, int ty);
void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows, int tx, int ty);

void eval(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int NumberOfColsNoMar, int NumberOfRowsNoMar,int tx, int ty);

// TODO: make sure that totalVBCols includes the margin, so these calculations are ok.
inline byte* getBordersVBfromXY(byte *fullBordersArry,int VBx,int VBy,int totalVBCols,int totalCols,int totalRows)
{
	return &(fullBordersArry[((VBy*totalVBCols)+VBx)*  (   (totalCols-MARGIN_SIZE_COLS)  *2 +  (totalRows-MARGIN_SIZE_ROWS)*2  )  ]);
}

void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows,int tx, int ty);

#endif

