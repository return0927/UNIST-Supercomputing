#include <stdio.h>
#include <math.h>
#define N 1000000
// function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int i;
  i = blockDim.x * blockIdx.x + threadIdx.x ;
  if (i<n) {
  	y[i] = x[i] + y[i];
  }
}
int main(void)
{
  int i;
  float maxError = 0.0f;
  float x[N], y[N], *d_x, *d_y;
  for (i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  cudaMalloc(&d_x, N*sizeof(float)); cudaMalloc(&d_y, N*sizeof(float));
  cudaMemcpy(d_x, &x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &y, N*sizeof(float), cudaMemcpyHostToDevice);
  // Run kernel on the elements on the GPU
  add<<<N/256+1,256>>>(N, d_x, d_y);
  cudaMemcpy(&y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x); cudaFree(d_y);
  // Check for errors (all values should be 3.0f)
  for (i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  printf("Max error: %f\n", maxError);
}
