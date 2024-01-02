CC = gcc
CFLAG = -O3 -g
LIBS = -L/usr/local/lib -lclBLAS -lOpenCL -lgfortran
INCLUDE = -I./
test : test.o
	$(CC) $(CFLAG) $(INCLUDE) -o test test.o cblas_LINUX.a blas_LINUX.a $(LIBS)
 
test.o : test.c
	$(CC) $(CFLAG) $(INCLUDE) -c test.c

clean:
	rm -f *.o
	rm -f test
	rm -rf .shaders
