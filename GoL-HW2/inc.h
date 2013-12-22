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

using namespace std;

#define NUM_BLOCKS_X 2
#define NUM_BLOCKS_Y 1
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 30

void kernel(
		byte* d_in,
		byte* d_out,
		int sizeX,
		int sizeY,
		int iterations,
		int* blockGenerations
		);

int host(int sizeX, int sizeY, byte* input, byte* output, int iterations, string outfilename);

#endif

