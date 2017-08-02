#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "sho.h"
#include <mpi.h>

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

int main(int argc, char **argv)
{
    double *result = malloc(sizeof(double) * N);
    
    double start = get_time();
    
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    int i_start[num_procs];
    int i_end[num_procs];
    int number_of_data[num_procs];
    int unit = (N + num_procs - 1) / num_procs;
    for (int rank = 0; rank < num_procs; rank++) {
        i_start[rank] = unit*rank;
        i_end[rank] = (rank == num_procs-1) ? N-1 : unit*(rank+1)-1;
        number_of_data[rank] = i_end[rank] - i_start[rank] +1;
    }
    
    run(i_start[my_rank], i_end[my_rank], result);
    
    if (0 == my_rank) {
        for (int rank = 1; rank < num_procs; rank++) {
            MPI_Recv(result+i_start[rank], number_of_data[rank], MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(result+i_start[my_rank], number_of_data[my_rank], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();

    if (0 == my_rank) {
        printf("Elapsed time: %fs\n", get_time() - start);

        double total = 0.;
        for(int i = 0; i < N; i++){
            total += result[i] / N;
        }
        printf("Total points: %d\n", N);
        printf("Mean eneregy: %f \n", total);
    }
    
    free(result);

    return 0;
}
