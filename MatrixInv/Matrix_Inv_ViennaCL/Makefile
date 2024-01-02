CC = g++
CFLAG = -O3 -g
LIBS = -lOpenCL
INCLUDE = -I./
test : matrixinv.o
	$(CC) $(CFLAG) $(INCLUDE) -o test matrixinv.o $(LIBS)
 
matrixinv.o : matrixinv.cpp
	$(CC) $(CFLAG) $(INCLUDE) -c matrixinv.cpp

clean:
	rm -f *.o
	rm -f test
	rm -f matrixinv
	rm -rf .shaders















