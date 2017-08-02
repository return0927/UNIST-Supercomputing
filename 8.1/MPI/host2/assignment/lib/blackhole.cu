#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "blackhole_lab.h"

#define THREAD_PER_BLOCK 128

__global__ void kernel(const int number_of_points /* add argments*/
		) {
    // add code for kernel
}

extern "C"
void extern_run(const int number_of_points
                /* add arguments */) 
{
    int blocks;
    // add code to init blocks, launch kernel and others
}


