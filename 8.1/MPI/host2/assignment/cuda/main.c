#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "blackhole_lab.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

void extern_run(const int number_of_points, double *d_value, bool *d_completed, bool *d_changed);

void run(int w_start, int w_end, int h_start, int h_end, int *status, double *y, double *z)
{
    double *d_value;
    bool *d_completed;
    bool *d_changed;

    int w_interval = w_end - w_start + 1;
    int h_interval = h_end - h_start + 1;
    int number_of_points = w_interval * h_interval;
    
    double *value_storage = malloc(sizeof(double) * number_of_points * C_NUMBER_OF_FUNCTIONS);
    bool    *completed = malloc(sizeof(bool) * number_of_points * C_NUMBER_OF_FUNCTIONS);
    //bool    *changed = malloc(sizeof(bool));

    fprintf(stderr, " RUN variables set");

    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_FUNCTIONS;
        int h = h_start + i / w_interval;
        int w = w_start + i % w_interval;
        // set initial value
        value[X_r    ] = INITIAL_X_r    (w, h, value);
        value[X_theta] = INITIAL_X_theta(w, h, value);
        value[X_phi  ] = INITIAL_X_phi  (w, h, value);
        value[U_r    ] = INITIAL_U_r    (w, h, value);
        value[U_theta] = INITIAL_U_theta(w, h, value);
        value[U_phi  ] = INITIAL_U_phi  (w, h, value);
        value[U_t    ] = INITIAL_U_t    (w, h, value);
        //printf("%5d", i);
        if(i == number_of_points-1) printf("!");
    }

    fprintf(stderr, " RAM Allocation 1");

    cudaMalloc((void **)&d_value, sizeof(double)*number_of_points*C_NUMBER_OF_FUNCTIONS); fprintf(stderr, ".");
    cudaMemcpy(d_value, value_storage, sizeof(double)*number_of_points*C_NUMBER_OF_FUNCTIONS, cudaMemcpyHostToDevice);

    fprintf(stderr, " 2");

    cudaMalloc((void **)&d_completed, sizeof(bool)*number_of_points*C_NUMBER_OF_FUNCTIONS);fprintf(stderr, ".");
    cudaMemcpy(d_completed, completed, sizeof(bool)*number_of_points*C_NUMBER_OF_FUNCTIONS, cudaMemcpyHostToDevice);

    fprintf(stderr, " 3\n");
    cudaMalloc((void **)&d_changed, sizeof(bool));printf(".\n");

   for (int k=0; k< C_TOTAL_STEP; k++) {
       printf(" FOR %d\t\t", k);
       bool changed = false;
       cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice);
       extern_run(number_of_points, d_value, d_completed, d_changed);
       cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
       if(!changed) break;
   }
    cudaMemcpy(value_storage, d_value, sizeof(double)*number_of_points*C_NUMBER_OF_FUNCTIONS, cudaMemcpyDeviceToHost);
    cudaMemcpy(completed, d_completed, sizeof(bool)*number_of_points*C_NUMBER_OF_FUNCTIONS, cudaMemcpyDeviceToHost);

    cudaFree(d_value);
    cudaFree(d_completed);
    cudaFree(d_changed);

    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_FUNCTIONS;
        int index = (h_start + i / w_interval) * C_RESOLUTION_WIDTH + (w_start + i % w_interval);
        // information return
        if (BOOL_CROSS_PICTURE(value)) {
            status[index] = HIT;
            y[index] = Y(value) / C_l;
            z[index] = Z(value) / C_l;
        } else if (BOOL_NEAR_HORIZON(value)) {
            status[index] = FALL;
        } else if (BOOL_OUTSIDE_BOUNDARY(value)) {
            status[index] = OUTSIDE;
        } else {
            status[index] = YET;
        }
    }
    
    free(value_storage);
}

void export(int *status, double *y, double *z)
{
    FILE *data_fp = fopen(C_OUTPUT_FILENAME, "wb");
    
    int resolution[2] = {C_RESOLUTION_WIDTH, C_RESOLUTION_HEIGHT};
    fwrite(resolution, sizeof(int), 2, data_fp);
    
    fwrite(status, sizeof(int   ), C_RESOLUTION_TOTAL_PIXELS, data_fp);
    fwrite(     y, sizeof(double), C_RESOLUTION_TOTAL_PIXELS, data_fp);
    fwrite(     z, sizeof(double), C_RESOLUTION_TOTAL_PIXELS, data_fp);
    
    fclose(data_fp);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.e-6;
}

int main() {
    printf(" Program started\n");
    int *status = malloc(sizeof(int   ) * C_RESOLUTION_TOTAL_PIXELS);
    double   *y = malloc(sizeof(double) * C_RESOLUTION_TOTAL_PIXELS);
    double   *z = malloc(sizeof(double) * C_RESOLUTION_TOTAL_PIXELS);

    int w_start = 0;
    int w_end   = C_RESOLUTION_WIDTH  - 1;
    int h_start = 0;
    int h_end   = C_RESOLUTION_HEIGHT - 1;
    
    double start = get_time();
    printf(" Running CUDA Function\n");
    run(w_start, w_end, h_start, h_end, status, y, z);
    printf("Elapsed time: %fs\n", get_time() - start);

    export(status, y, z);
    
    free(status);
    free(y);
    free(z);
    
    return 0;
}
