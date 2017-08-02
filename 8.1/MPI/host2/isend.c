#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){

    int my_rank, num_proc;

    // MPI Initialization : INIT, RANK, PROCESSORS
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // NODE01 : 0 | NODE02 : 1
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc); // VAL 2 (Node count)

    // Sending to LEFT
    int my_answer[5] = {0};
    int left = (my_rank == 0) ? num_proc-1 : my_rank-1; // NODE01(HOST01): 1 | NODE02(HOST02) : 0

    MPI_Request request;
    MPI_Isend(my_answer, 5, MPI_INT, left, 4, MPI_COMM_WORLD, &request);

    // Receiving from RIGHT
    int others_answer[5] = {};
    int right = (my_rank == num_proc-1) ? 0: my_rank+=1;

    MPI_Recv(others_answer, 5, MPI_INT, right, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    return 0;
}