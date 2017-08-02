#include <stdio.h>
#include <math.h>

#define N 1000000

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  int i;
  for (i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int i;
  float maxError = 0.0f;
  float x[N];
  float y[N];

  // initialize x and y arrays on the host
  for (i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  for (i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  printf("Max error: %f\n", maxError);

  return 0;
}
