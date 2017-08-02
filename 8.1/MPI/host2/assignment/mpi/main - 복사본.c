#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "blackhole_lab.h"
#include <mpi.h>

#define MCW MPI_COMM_WORLD
#define MSI MPI_STATUS_IGNORE

void ray_trace(const int number_of_points, double *value_storage, bool *completed, bool *changed) {
#define C_RK_ORDER 4
    const double RK_factor1[] = {1. / 2.};
    const double RK_factor2[] = {0., 1. / 2.};
    const double RK_factor3[] = {0., 0., 1.};
    const double RK_factor4[] = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};
    const double *const RK_factor[C_RK_ORDER] = {RK_factor1, RK_factor2, RK_factor3, RK_factor4};

// time integration
    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_FUNCTIONS;
        if (!completed[i]) {
            completed[i] = BOOL_STOP_CONDITION(value);
        }
        if (!completed[i]) {
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
                    for (int j = 0; j < C_NUMBER_OF_FUNCTIONS; j++) {
                        value[j] = value_temp[j];
                    }
                } else {
                    derivative[k][X_r] = DERIVATIVE_X_r    (value_temp);
                    derivative[k][X_theta] = DERIVATIVE_X_theta(value_temp);
                    derivative[k][X_phi] = DERIVATIVE_X_phi  (value_temp);
                    derivative[k][U_t] = DERIVATIVE_U_t    (value_temp);
                    derivative[k][U_r] = DERIVATIVE_U_r    (value_temp);
                    derivative[k][U_theta] = DERIVATIVE_U_theta(value_temp);
                    derivative[k][U_phi] = DERIVATIVE_U_phi  (value_temp);
                }
            }
            *changed = true;
        }
    }
}

/*
*  w_start     Image Width start
*  w_end       ``
*
*  h_start     Image Height start
*  h_end       ``
*
*  status      Reach, Escape, Dropped, Yet
*
*  *y      Start X Coordination
*  *z      Start Y Coordination
*/
void run(int w_start, int w_end, int h_start, int h_end, int *status, double *y, double *z) {
    int w_interval = w_end - w_start + 1;
    int h_interval = h_end - h_start + 1;
    int number_of_points = w_interval * h_interval;
//printf("%d\n",number_of_points);

    double *value_storage = malloc(sizeof(double) * number_of_points * C_NUMBER_OF_FUNCTIONS);
    bool *completed = malloc(sizeof(bool) * number_of_points);

    for (int i = 0; i < number_of_points; i++) {
        double *value = value_storage + i * C_NUMBER_OF_FUNCTIONS;
        int h = h_start + i / w_interval;
        int w = w_start + i % w_interval;
// set initial value
        value[X_r] = INITIAL_X_r    (w, h, value);
        value[X_theta] = INITIAL_X_theta(w, h, value);
        value[X_phi] = INITIAL_X_phi  (w, h, value);
        value[U_r] = INITIAL_U_r    (w, h, value);
        value[U_theta] = INITIAL_U_theta(w, h, value);
        value[U_phi] = INITIAL_U_phi  (w, h, value);
        value[U_t] = INITIAL_U_t    (w, h, value);

        completed[i] = false;
    }

    for (int k = 0; k < C_TOTAL_STEP; k++) {
        bool changed = false;
        ray_trace(number_of_points, value_storage, completed, &changed);
        if (!changed) break;
    }

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


// Export image
void export(int *status, double *y, double *z) {
    FILE *data_fp = fopen(C_OUTPUT_FILENAME, "wb");

    int resolution[2] = {C_RESOLUTION_WIDTH, C_RESOLUTION_HEIGHT};
    fwrite(resolution, sizeof(int), 2, data_fp);

    fwrite(status, sizeof(int), C_RESOLUTION_TOTAL_PIXELS, data_fp);
    fwrite(y, sizeof(double), C_RESOLUTION_TOTAL_PIXELS, data_fp);
    fwrite(z, sizeof(double), C_RESOLUTION_TOTAL_PIXELS, data_fp);

    fclose(data_fp);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.e-6;
}

int main(int argc, char **argv) {
    int *status = malloc(sizeof(int) * C_RESOLUTION_TOTAL_PIXELS);
    double *y = malloc(sizeof(double) * C_RESOLUTION_TOTAL_PIXELS);
    double *z = malloc(sizeof(double) * C_RESOLUTION_TOTAL_PIXELS);

    int w_start = 0;
    int w_end = C_RESOLUTION_WIDTH - 1;
    int h_start = 0;
    int h_end = C_RESOLUTION_HEIGHT - 1;

//run(w_start, w_end, h_start, h_end, status, y, z);

// MPI Initialization
    int my_rank, num_proc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &my_rank);
    MPI_Comm_size(MCW, &num_proc);

    double start = get_time();

    /*int startPixelHeight[num_proc], startPixelWidth[num_proc];
    int endPixelHeight[num_proc], endPixelWidth[num_proc];
    int chunkPixelCount[num_proc];
    int unitHeight = C_RESOLUTION_HEIGHT/num_proc,
            unitWidth = C_RESOLUTION_WIDTH/num_proc;

    //printf("%d %d\n",unitHeight, unitWidth);
    for (int rank = 0; rank < num_proc; rank++) {
        startPixelHeight[rank] = unitHeight*rank;
        endPixelHeight[rank] = (rank == num_proc-1) ? C_RESOLUTION_HEIGHT -1 : unitHeight*(rank+1)-1;

        startPixelWidth[rank] = 0;
        endPixelWidth[rank] = C_RESOLUTION_WIDTH-1;
        //startPixelWidth[rank] = (int)rank/num_proc*unitWidth;
        //endPixelWidth[rank] = (rank == num_proc-1) ? C_RESOLUTION_WIDTH -1: unitWidth*rank;

        chunkPixelCount[rank] = (endPixelHeight[rank]-startPixelHeight[rank]+1)*(endPixelWidth[rank]-startPixelWidth[rank]+1);
    }

    printf("host%2d : sH %3d\teH %3d\tsW %3d\teW %3d\n", my_rank, startPixelHeight[my_rank], endPixelHeight[my_rank], startPixelWidth[my_rank], endPixelWidth[my_rank]);

    run(startPixelHeight[my_rank], endPixelHeight[my_rank], startPixelWidth[my_rank], endPixelWidth[my_rank], status, y, z);

    if (my_rank == 0){
        // Master
        for(int i=1; i<num_proc; i++){
            MPI_Recv(status+startPixelHeight)
        }

    }*/

    int i_start[num_proc];
    int i_end[num_proc];
    int number_of_data[num_proc];
    int unit = (C_RESOLUTION_TOTAL_PIXELS + num_proc - 1) / num_proc;
    for (int rank = 0; rank < num_proc; rank++) {
        i_start[rank] = unit*rank;
        i_end[rank] = (rank == num_proc-1) ? C_RESOLUTION_TOTAL_PIXELS-1 : unit*(rank+1)-1;
        number_of_data[rank] = i_end[rank] - i_start[rank] +1;
    }

    int myHeightStart = (my_rank == 0) ? 0 : (int)i_start[my_rank]/100;
    int myHeightEnd = (my_rank == num_proc-1) ? C_RESOLUTION_HEIGHT-1 : (int)i_end[my_rank]/100-5;

    //printf("host%2d : sH %2d\t\teH %2d\n", my_rank, (my_rank == 0) ? 0 : (int)i_start[my_rank]/100, (my_rank == num_proc-1) ? C_RESOLUTION_HEIGHT-1 : (int)i_end[my_rank]/100);
    run(0, C_RESOLUTION_WIDTH-1, myHeightStart, myHeightEnd, status, y, z);

    int rankHeightStart, rankHeightEnd;
    if (my_rank == 0) {
        for (int rank=1; rank<num_proc; rank++) {
            rankHeightStart = (rank == 0) ? 0 : (int)i_start[rank]/100;
            rankHeightEnd = (rank == num_proc-1) ? C_RESOLUTION_HEIGHT-1 : (int)i_end[rank]/100-5;

            printf("HOST%d : %d", rank, (int)status +myHeightStart*C_RESOLUTION_WIDTH);
            MPI_Recv(status +myHeightStart*C_RESOLUTION_WIDTH, (rankHeightEnd-rankHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, rank, 0, MCW, MSI);
            MPI_Recv(y +myHeightStart*C_RESOLUTION_WIDTH, (rankHeightEnd-rankHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, rank, 1, MCW, MSI);
            MPI_Recv(z +myHeightStart*C_RESOLUTION_WIDTH, (rankHeightEnd-rankHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, rank, 2, MCW, MSI);
        }
    } else {
        MPI_Send(status+ myHeightStart*C_RESOLUTION_WIDTH, (myHeightEnd-myHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, 0, 0, MCW);
        MPI_Send(y+ myHeightStart*C_RESOLUTION_WIDTH, (myHeightEnd-myHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, 0, 1, MCW);
        MPI_Send(z+ myHeightStart*C_RESOLUTION_WIDTH, (myHeightEnd-myHeightStart+1)*C_RESOLUTION_WIDTH, MPI_INT, 0, 2, MCW);
    }


    MPI_Finalize();
    if (my_rank == 0) {
        printf("Elapsed time: %fs\n", get_time() - start);
        printf("Number: %d\n",num_proc);
        printf("num_of_data: %d\n",number_of_data[0]);

        export(status, y, z);
    } else {
    }

    free(status);
    free(y);
    free(z);

    return 0;
}
