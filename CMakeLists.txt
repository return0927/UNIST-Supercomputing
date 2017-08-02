cmake_minimum_required(VERSION 3.8)
project(Test)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c 8.1/MPI/mpi.c 8.1/MPI/host2/isend.cpp 8.1/MPI/host2/gpu_programming/cdau.c.cpp 8.1/MPI/host2/gpu_programming/gpu.c)
add_executable(Test ${SOURCE_FILES})