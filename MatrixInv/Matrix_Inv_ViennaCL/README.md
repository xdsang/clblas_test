## 算子依赖ViennaCL和Boost的环境, 编译ViennaCL步骤：
1. 安装boost
apt install libboost-all-dev  
安装完成后可能在cmake时出席那lib和head file 找不到的情况，需要设置一下环境变量，如下：  
export BOOST_LIBRARYDIR=/usr/lib/x86_64-linux-gnu/      
/usr/lib/x86_64-linux-gnu/是我的boost库安装的路径，不同的系统或cpu架构路径可能会不同，可通过find / -name libboost* 找到具体路径  
export BOOST_INCLUDEDIR=/usr/include/      
/usr/include/ 是boost头文件所在的地方，可通过 find / -name version.hpp 找到boost头文件具体所在目录

2. build  ViennaCL 
ViennaCL下载地址 http://confluence.srv/download/attachments/77700518/ViennaCL-1.7.1.tar.gz?version=2&modificationDate=1687334026605&api=v2    
tar -zxvf ViennaCL-1.7.1.tar.gz  
cd ViennaCL-1.7.1  
mkdir build  &&  cd build   
cmake -DBUILD_TESTING=ON  ../   
make -j 12   
make install   

## 算子编译运行示例
【编译】
make clean   
make   

【运行】
正式运行前可以在终端  ./test --help查看参数含义   
示例: ./test -n 256   
        @ -n 256 表示矩阵尺寸是256*256，也可以不设置，不设置n默认为256   
        
【示例】  
./test -n 128 （尺寸过小，CPU计算性能会优于GPU）   
./test -n 256   
./test -n 512   
./test -n 1024   

