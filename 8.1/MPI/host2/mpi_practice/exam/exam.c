#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // TEACHER
    if (0 == my_rank) {
        printf("Teacher: Send exam paper\n");

        // 시험지 전송
        // 시험지 태그 1
        char question[5][10] = {"Q1", "Q2", "Q3", "Q4", "Q5"};
        for (int rank = 1; rank < num_procs; rank++) { // 전체 노드
            MPI_Send(question, 5*10, MPI_CHAR, rank, 1, MPI_COMM_WORLD);
        }
        
        printf("Teacher: Receive exam completion signs\n");

        // 반환시그널 확인
        // 시그널 태그 2
        for (int rank = 1; rank < num_procs; rank++) {
            MPI_Recv(NULL, 0, MPI_INT, rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        printf("Teacher: Send rotation signs\n");

        // 답안지 교환 시그널 전송
        // 태그 3
        for (int rank = 1; rank < num_procs; rank++) {
            MPI_Send(NULL, 0, MPI_INT, rank, 3, MPI_COMM_WORLD);
        }

        printf("Teacher: Receive rotation completion signs\n");

        // 답안지 교환 완료 시그널 수신
        // 태그 5
        for (int rank = 1; rank < num_procs; rank++) {
            MPI_Recv(NULL, 0, MPI_INT, rank, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        printf("Teacher: Send correct answers\n");

        // 답안지 전송
        // 태그 6
        int correct_answer[5] = {4, 2, 3, 1, 5};
        for (int rank = 1; rank < num_procs; rank++) {
            MPI_Send(correct_answer, 5, MPI_INT, rank, 6, MPI_COMM_WORLD);
        }
    }

    // STUDENT
    else {
        // 시험지 수신
        // 태그 1
        char question[5][10];
        MPI_Recv(question, 5*10, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 내 답안 생성
        int my_answer[5];
        for (int i = 0; i < 5; i++) {
            my_answer[i] = (my_rank + i) % 5 + 1;
        }

        // 반환시그널 전송
        // 태그 2
        MPI_Send(NULL, 0, MPI_INT, 0, 2, MPI_COMM_WORLD);

        // 답안지 교환 시그널 수신
        // 태그 3
        MPI_Recv(NULL, 0, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 내 답안지 교환
        // 태그 4
        int left = (my_rank == 1) ? num_procs - 1 : my_rank - 1;
        MPI_Request request;
        MPI_Isend(my_answer, 5, MPI_INT, left, 4, MPI_COMM_WORLD, &request);

        // 답안지 수신
        // 태그 4
        int right = (my_rank == num_procs - 1) ? 1 : my_rank + 1;
        int other_answer[5];
        MPI_Recv(other_answer, 5, MPI_INT, right, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // 답안지 교환 시그널 전송
        // 태그 5
        MPI_Send(NULL, 0, MPI_INT, 0, 5, MPI_COMM_WORLD);
        
        // 답안지 전송 시그널 수신
        // 태그 6
        int correct_answer[5];
        MPI_Recv(correct_answer, 5, MPI_INT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Scoring
        bool other_score[5] = {false};
        for (int i = 0; i < 5; i++) {
            if (correct_answer[i] == other_answer[i]) other_score[i] = true;
        }
        
        // 다음 노드에 채점결과 전송
        // 태그 7
        MPI_Isend(other_answer, 5, MPI_C_BOOL, right, 7, MPI_COMM_WORLD, &request);
        
        // 이전 노드에서 채점결과 수신
        bool my_score[5];
        MPI_Recv(my_score, 5, MPI_C_BOOL, left, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        
        printf("Student %d: ", my_rank);
        for (int i = 0; i < 5; i++) {
            if (my_score[i])
                printf("correct ");
            else
                printf("wrong ");
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
