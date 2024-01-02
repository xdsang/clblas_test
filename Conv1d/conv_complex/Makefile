CC = gcc
CFLAG = -O3 -g
LIBS = -lOpenCL
INCLUDE = -I./
test : test.o
	$(CC) $(CFLAG) $(INCLUDE) -o test test.o $(LIBS)

test.o : test.c
	$(CC) $(CFLAG) $(INCLUDE) -c test.c

clean:
	rm -f *.o
	rm -f test
	rm -f conv1
	rm -rf .shaders















