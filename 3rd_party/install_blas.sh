#!/bin/bash

sudo apt-get install gfortran 

tar -xvf blas-3.10.0.tgz
cd BLAS-3.10.0/
make -j12
gfortran -c -O3 -fPIC *.f
ar rv libblas.a *.o
cp libblas.a  /usr/local/lib/

cd ..

tar -xvf cblas.tgz
cd CBLAS/
mv Makefile.LINUX Makefile.in
cp ../BLAS-3.10.0/libblas.a  ./testing
make -j12
cp lib/cblas_LINUX.a  /usr/local/lib/

cd ..

unzip clBLAS-master.zip
cd clBLAS-master
cd src/
mkdir build
cd build
cmake ..
make -j12
cp ./library/libclBLAS.so*  /usr/local/lib/
