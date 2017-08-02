#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "blackhole_lab.h"

#define THREAD_PER_BLOCK 128

__global__ void kernel(const int number_of_points, double *d_value, bool *d_completed, bool *d_changed) {
    #define C_RK_ORDER 4
    const double RK_factor1[] = {1. / 2.                           };
    const double RK_factor2[] = {     0., 1. / 2.                  };
    const double RK_factor3[] = {     0.,      0.,      1.         };
    const double RK_factor4[] = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};
    const double * const RK_factor[C_RK_ORDER] = {RK_factor1, RK_factor2, RK_factor3, RK_factor4};

    printf(" KERNEL ");
    // time integration
    for (int i = 0; i < number_of_points; i++) {
        printf(".");
        double *value = d_value + i * C_NUMBER_OF_FUNCTIONS;
        if (!d_completed[i]) {
            d_completed[i] = BOOL_STOP_CONDITION(value);
        }
        if (!d_completed[i]) {
            double value_temp[C_NUMBER_OF_FUNCTIONS];
            double derivative[C_RK_ORDER][C_NUMBER_OF_FUNCTIONS];
            for (int k = 0; k <= C_RK_ORDER; k++) {
                for (int j = 0; j < C_NUMBER_OF_FUNCTIONS; j++) {
                    value_temp[j] = value[j];
                }
                for (int l = 0; l < k; l++) {
                    for (int j = 0; j < C_NUMBER_OF_FUNCTIONS; j++) {
                        value_temp[j] += RK_factor[k - 1][l] * derivative[l][j] * C_DELTA_TIME;
                    }
                }
                if (C_RK_ORDER == k) {
                    for (int j = 0 ;j < C_NUMBER_OF_FUNCTIONS; j++) {
                        value[j] = value_temp[j];
                    }
                } else {
                    derivative[k][X_r    ] = DERIVATIVE_X_r    (value_temp);
                    derivative[k][X_theta] = DERIVATIVE_X_theta(value_temp);
                    derivative[k][X_phi  ] = DERIVATIVE_X_phi  (value_temp);
                    derivative[k][U_t    ] = DERIVATIVE_U_t    (value_temp);
                    derivative[k][U_r    ] = DERIVATIVE_U_r    (value_temp);
                    derivative[k][U_theta] = DERIVATIVE_U_theta(value_temp);
                    derivative[k][U_phi  ] = DERIVATIVE_U_phi  (value_temp);
                }
            }
            *d_changed = true;
        }
    }

    printf("\n");

}






extern "C"
void extern_run(const int number_of_points, double *d_value, bool *d_completed, bool *d_changed) {
    int blocks = number_of_points/THREAD_PER_BLOCK +1;

    // add code to init blocks, launch kernel and others
    printf(" C ");
    kernel<<< blocks, THREAD_PER_BLOCK >>> (number_of_points, d_value, d_completed, d_changed);
}


