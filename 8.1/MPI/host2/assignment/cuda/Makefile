C    	= gcc
TARGET = bhlab
NCFLAGS = -O3 $(MACRO) $(INC)
CFLAGS  = -O3 -std=c99 $(MACRO) $(INC)
OBJS 	= blackhole.o
LIBS	:= -lm -lcudart -L/usr/local/cuda/lib
INC     := -I/usr/local/cuda/include
#KERR = -DKERR_SPACETIME
#WIDTH = -DC_RESOLUTION_WIDTH=400
#HEIGHT = -DC_RESOLUTION_HEIGHT=400
MACRO = $(KERR) $(WIDTH) $(HEIGHT)

all	: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o bhlab main.c $(OBJS) $(LIBS)

clean :
	rm -rf *.o $(TARGET)

.SUFFIXES : .cu .o

.cu.o:   ; nvcc -c $(NCFLAGS) -gencode arch=compute_32,code=sm_32  $?


