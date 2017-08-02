CC = gcc
TARGET = energy
OBJECTS = main.o energy.o 
CFLAGS = -O3 $(INC)
NVCFLAGS = -O3
LINK = -lm -lcudart -L/usr/local/cuda/lib -lstdc++ 
INC = -I/usr/local/cuda/include

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LINK)

clean :
	rm $(TARGET) $(OBJECTS)

.SUFFIXES : .cu .o

.cu.o:	
	nvcc -c $(NVCFLAGS) -gencode arch=compute_32,code=sm_32 $?
