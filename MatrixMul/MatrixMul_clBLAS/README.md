## 算子依赖clBLAS
依赖的源文件下载地址  http://confluence.srv/download/attachments/77700518/clBLAS_install.zip?version=1&modificationDate=1686657143422&api=v2   
在编译kernel之前先解压clBLAS_install.zip 压缩包编译其中的源码文件，   
根据压缩包中的BlasConf.txt的说明进行安装。    
编译好之后将clBLAS_install/CBLAS/lib/cblas_LINUX.a和clBLAS_install/BLAS-3.10.0/blas_LINUX.a 两个文件一起复制到MatrixMul/MatrixMul_clBLAS/clblasCgemm 和 MatrixMul/MatrixMul_clBLAS/clblasSgemm     

## 算子编译和运行示例
【功能】
clblasSgemm——实现的是浮点实矩阵乘法（调用clBLAS库，运行前必须先安装clBLAS环境）  
clblasCgemm——实现的是浮点复矩阵乘法（调用clBLAS库，运行前必须先安装clBLAS环境）
   
【编译】   
make clean   
make    
    
【运行】   
正式运行前可以在终端  ./test --help查看参数含义   
示例: ./test -n 4 -b 1 -p 1   
        @ -n 4 表示设置矩阵尺寸为4*4，可以不设置，不设置默认输入为16*16的方阵    
        @ -b 1 表示设置batch模式，可以不设置，不设置默认为1   
        @ -p 1 表示开启打印输入输出数据，可以不设置，不设置默认为0，即不在终端打印输入输出结果    

【高负载pdump运行】    
./test -n 512       

