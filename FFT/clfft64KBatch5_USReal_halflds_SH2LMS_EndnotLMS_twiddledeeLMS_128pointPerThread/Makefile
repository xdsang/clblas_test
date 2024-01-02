CC = g++
CFLAG = -O3 -g
LIBS = -lOpenCL
INCLUDE = -I./
test : test.o
	$(CC) $(CFLAG) $(INCLUDE) -o test test.o $(LIBS)
 
test.o : test.cpp
	$(CC) $(CFLAG) $(INCLUDE) -c test.cpp

clean:
	rm -f *.o
	rm -f test
	rm -rf .shaders















