#include "energy.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#define THREAD_PER_BLOCK 128

__global__ void kernel(int number_of_points, double *d_value)
{
    int i;
    i = blockDim.x * blockIdx.x + threadIdx.x ;
    if (i<number_of_points) {
        double *value = d_value + i * C_NUMBER_OF_VARIABLES;
        double x = C_A * sin(d_value[phase]);           // displacement
        double v = C_A * C_omega * cos(d_value[phase]);   // velocity
        double U = C_k * x * x / 2.;                      // potential energy
        double K = C_m * v * v / 2.;                      // kinetic energy
        value[E] += (U + K) / M;                    // mechanical energy
        value[phase] += C_omega;                    // phase progress
    }
}

extern "C"
void extern_run(const int number_of_points, double *d_value)
{
    int blocks = number_of_points/THREAD_PER_BLOCK +1;
    // add code to init blocks
    kernel<<<blocks, THREAD_PER_BLOCK>>>(number_of_points, d_value);
}
