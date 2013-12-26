#ifndef __INC__

#define __INC__

#define MEASUREMENTS

#include "FieldReader.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

__global__ void kernel(
	byte* input,
	byte* output,
	const int numberOfRows,
	const int numberOfCols,
	int numberOfVirtualBlockX,
	int numberOfVirtualBlockY,
	int iterations,
	byte *bordersArray,
	byte *bordersArray2,
	int * blockGenerations
);

byte* host(int sizeX, int sizeY, byte* input, int iterations);


#define NUM_BLOCKS_X 8
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

#define WARPS_FOR_BORDERS 4

#define WARPS_FOR_PACKING 4

#define NUMBER_OF_COLS 1000
#define NUMBER_OF_ROWS 1000


__global__ void kernel(byte* input, byte* output,const int numberOfRows,const int numberOfCols,
	int numberOfVirtualBlockX, int numberOfVirtualBlockY,
	int iterations,byte *bordersArray,int * blockGenerations);

__forceinline__ __device__ byte * getUPBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols)*0]);
}

__forceinline__ __device__ byte * getDOWNBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols)*1]);
}

__forceinline__ __device__ byte * getLEFTBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols)*2]);
}

__forceinline__ __device__ byte * getRIGHTBorder(byte * BordersAryPlace,int totalCols,int totalRows)
{
	return &(BordersAryPlace[(totalCols)*2 + (totalRows)*1]);
}


__forceinline__ __device__ void  fillBorders(byte * blockWithMargin,byte *fullBordersArry,int VBx,int VBy,int totalVBCols,
	int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows);

__forceinline__ __device__ void packer(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows);

__forceinline__ __device__ void unpacker(byte* in, byte* out, int numUsedCols, int numUsedRows, int numTotalCols, int numTotalRows);

__forceinline__ __device__ void eval(byte * srcBlockWithMargin,byte * tarBlockWithMargin,int NumberOfColsNoMar, int NumberOfRowsNoMar);

__forceinline__ __device__ byte* getBordersVBfromXY(byte *fullBordersArry,int VBx,int VBy,int totalVBCols,int totalCols,int totalRows)
{
	return &(fullBordersArry[(((VBy+1)*(totalVBCols+VB_MARGIN_SIZE))+VBx+1)*  (   (totalCols)  *2 +  (totalRows)*2  )  ]);
}

__forceinline__ __device__ void share2glob(byte * blockWithMargin,byte *BordersAryPlace,int usedColsNoMar, int usedRowsNoMar, int totalCols,int totalRows);

#endif
