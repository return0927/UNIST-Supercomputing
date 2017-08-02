#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "energy.h"

#define N 10000

void get_energy(int number_of_points, double *value_storage)
{
    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_VARIABLES;
        double x = C_A*sin(value[phase]);           // displacement
        double v = C_A*C_omega*cos(value[phase]);   // velocity
        double U = C_k*x*x/2.;                      // potential energy
        double K = C_m*v*v/2.;                      // kinetic energy
        value[E] += (U + K) / N;                    // mechanical energy
        value[phase] += C_omega;                    // phase progress
    }
}

void run(int i_start, int i_end, double *result)
{
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
    
    for (int t = 0; t < N; t++) {
        get_energy(number_of_points, value_storage);
    }

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
    double *result = malloc(sizeof(double) * N);

    int i_start = 0;
    int i_end = N - 1;
    
    double start = get_time();
    run(i_start, i_end, result);
    printf("Elapsed time: %fs\n", get_time() - start);

    double total = 0.;
    for(int i = 0; i < N; i++){
        total += result[i];
    }
    total /= N;
    printf("Total points: %d\n", N);
    printf("Mean eneregy: %f \n", total);
    
    free(result);

    return 0;
}
