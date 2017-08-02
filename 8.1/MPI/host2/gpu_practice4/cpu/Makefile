CC = gcc
TARGET = energy
OBJECTS = main.o 
CFLAGS = -std=c99 -O1  -pg -g
LINK = -lm

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LINK)

clean :
	rm $(TARGET) $(OBJECTS)
