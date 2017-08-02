#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "energy.h"

#define THREAD_PER_BLOCK 128

__global__ void kernel(int number_of_points, double *d_value)
{
   // add code
  int i;
  i = blockDim.x * blockIdx.x + threadIdx.x ;
  if (i<number_of_points) {
        double *value = d_value + i * C_NUMBER_OF_VARIABLES;
        double x = C_A*sin(value[phase]);           // displacement
        double v = C_A*C_omega*cos(value[phase]);   // velocity
        double U = C_k*x*x/2.;                      // potential energy
        double K = C_m*v*v/2.;                      // kinetic energy
        value[E] += (U + K) / N;                    // mechanical energy
        value[phase] += C_omega;                    // phase progress
//        if (i==0) printf("%f\n", value[phase]);
  }
}

extern "C"
void extern_run(const int number_of_points, double *d_value)
{
    int blocks = (number_of_points % THREAD_PER_BLOCK == 0) ? number_of_points / THREAD_PER_BLOCK : number_of_points / THREAD_PER_BLOCK +1; 
    kernel<<<blocks, THREAD_PER_BLOCK>>>(number_of_points, d_value);
}
