C    	= mpicc
NCFLAGS = -O3 $(MACRO)
CFLAGS  = -O3 -std=c99 $(MACRO) 
OBJS 	= blackhole.o
LIBS	:= -lm -lcudart -L/usr/local/cuda/lib
INC     := -I/usr/local/cuda/include
#KERR = -DKERR_SPACETIME
#WIDTH = -DC_RESOLUTION_WIDTH=400
#HEIGHT = -DC_RESOLUTION_HEIGHT=400
MACRO = $(KERR) $(WIDTH) $(HEIGHT)

all	: mytest

mytest	: $(OBJS)
	$(C) $(CFLAGS) -o bhlab main.c $(INC) $(OBJS) $(LIBS)

clean :
	rm -rf *.o main data

.SUFFIXES : .cu .o

.cu.o:   ; nvcc -c $(NCFLAGS) -gencode arch=compute_32,code=sm_32  $?


