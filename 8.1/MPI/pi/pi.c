// KISTI educational PI calculation
// Copyright(c) 2012 KISTI Supercomputing Center
#include <stdio.h>
#include <math.h>

int main() {
    const long num_step=1000000000;
    long i;
    double pi = 0.0, step, x;

    step = (1.0/(double)num_step);
    printf("-----------------------------------------------------------\n");
#pragma omp parallel for private(x) reduction(+:pi)
    for(i=0; i<num_step; i++) {
        x = ((double)i - 0.5) * step;
        pi += 4.0/(1.0+x*x);
    }
    pi *= step;

    printf("PI = %.15lf (Error = %e)\n", pi, fabs(acos(-1.0)-pi));
    printf("-----------------------------------------------------------\n");
    return 0;
}
