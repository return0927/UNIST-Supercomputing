#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "energy.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

void extern_run(const int number_of_points, double *d_value);

void run(int i_start, int i_end, double *result)
{
    double *d_value;
    
    int number_of_points = i_end - i_start + 1;
    double *value_storage = malloc(sizeof(double) * number_of_points * C_NUMBER_OF_VARIABLES);
    
    for (int i = 0; i < number_of_points; i++) {
        result[i_start + i] = 0.;
    }
    
    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_VARIABLES;
        value[phase] = (double)(i_start + i);
        value[E] = 0.;
    }

    /*
    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_VARIABLES;
        value[phase] = (double)(i_start + i);
        value[E] = 0.;
    }
     */

    cudaMalloc(&d_value, sizeof(double)*number_of_points*C_NUMBER_OF_VARIABLES);
    cudaMemcpy(d_value, value_storage, sizeof(double)*number_of_points*C_NUMBER_OF_VARIABLES, cudaMemcpyHostToDevice);
    for(int i=0; i<number_of_points; i++){
        extern_run(number_of_points, d_value);
    }
    cudaMemcpy(value_storage, d_value, sizeof(double)*number_of_points*C_NUMBER_OF_VARIABLES, cudaMemcpyDeviceToHost);
    cudaFree(d_value);
 
    *result = 0.;
    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_VARIABLES;
        result[i_start + i] = value[E];
    }
    
    free(value_storage);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.e-6;
}

int main()
{
    double *result = malloc(sizeof(double) * M);
    
    int i_start = 0;
    int i_end = M - 1;
    
    double start = get_time();
    run(i_start, i_end, result);
    printf("Elapsed time: %fs\n", get_time() - start);
    
    double total = 0.;
    for(int i = 0; i < M; i++){
        total += result[i];
    }
    total /= M;
    printf("Total points: %d\n", M);
    printf("Mean eneregy: %f \n", total);
    
    free(result);
    
    return 0;
}

