/* ==================================================================================
* @Copyright    : 2023 by Innosilicon Technologies Ltd. All Rights Reserved
* @FileName     : test.cpp
* @Date         : May.2023
* @Author       : wusk
* @Description  : LU Factorization and Inversion of Square Matrix with float elements
====================================================================================== */
#define VIENNACL_WITH_OPENCL   // Must be set if you want to use ViennaCL algorithms on OpenCL device
#define VIENNACL_WITH_UBLAS     //Must be set if you want to use ViennaCL algorithms on ublas objects
//#define VIENNACL_BUILD_INFO
//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_CONTEXT
// #define PRINTF
//#define HANDWRITE

//System headers
#include <cmath>
#include <vector>
#include <iostream>
//uBLAS headers
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
//ViennaCL headers
#include <viennacl/scalar.hpp>              //GPU标量  
#include <viennacl/vector.hpp>              //GPU矢量
#include <viennacl/matrix.hpp>              //GPU矩阵
#include <viennacl/linalg/lu.hpp>           //LU substitution routines
#include <viennacl/ocl/backend.hpp>
#include <viennacl/linalg/prod.hpp>         //generic matrix-vector product
#include <viennacl/tools/timer.hpp>         //跨平台计时器
#include <viennacl/linalg/norm_2.hpp>       //generic l2-norm for vectors
#include <viennacl/linalg/direct_solve.hpp>

using namespace std;

typedef float ScalarType;
std::size_t N = 256;
double time0=0;
double time1=0;

double time2=0;
double time3=0;
double lu_time=0;
double inv_time=0;

static void print_help(const char* name, int ret)
{
    printf("Usage: [-n matrix rows/cols] \n");
    printf("-n matrix rows/cols: user defined, default n = 256, Matrix size is 256*256 \n");
    exit(ret);
}

char* loadFile(const char* file)
{
    FILE *program_handle;
    size_t program_size;
    char *program_buffer;

    program_handle = fopen(file, "r");
    if(program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);   
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_buffer;
}

void ShowMatrix(float *Mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f\t",Mat[i*cols+j]);
        }
        printf("\n");
    }
}

int main(int argc , char *argv[]) 
{
    for (size_t i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                N = atoi(argv[i]);
            }
        }
        else
        {
            print_help(argv[0], 1);
        }
    }

    cl_int err;
    viennacl::tools::timer timer;
    viennacl::ocl::platform pf;
    vector<viennacl::ocl::platform> pfs = viennacl::ocl::get_platforms();
    cout << pf.info() << endl;
    cl_uint num_platforms =  pfs.size();

	cl_platform_id *platform_ids = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	cl_uint *num_devices = (cl_uint *)malloc(num_platforms * sizeof(cl_uint));
	cl_device_id **device_ids = (cl_device_id **)malloc(num_platforms * sizeof(cl_device_id*));

	cl_context *contexts = (cl_context *)malloc(sizeof(cl_context) * num_platforms);
	cl_mem *mem_matrixs = (cl_mem *)malloc(sizeof(cl_mem) * num_platforms);
	cl_command_queue **cmd_queues = (cl_command_queue **)malloc(sizeof(cl_command_queue* ) * num_platforms);

    /*CPU memory*/
    std::size_t lu_dim = N;
    std::size_t matrix_size = lu_dim * lu_dim;
    boost::numeric::ublas::matrix<ScalarType> square_matrix(lu_dim, lu_dim);
    boost::numeric::ublas::matrix<ScalarType> square_matrix_cpu(lu_dim, lu_dim);
    boost::numeric::ublas::identity_matrix<ScalarType> E_(lu_dim);
    square_matrix = E_;
    square_matrix_cpu = square_matrix;

	for (size_t i = 0; i < num_platforms; ++i)
	{
		contexts[i] = 0;
		platform_ids[i] = 0;
		mem_matrixs[i] = 0;
		num_devices[i] = 0;
		cmd_queues[i] =  NULL;
		device_ids[i] =  NULL;
	}

	printf( "Total CL platforms: %d\n\n" , num_platforms);
    for (size_t i = 0; i< num_platforms; ++i)
    {
        viennacl::ocl::platform pf(i);
        cout << "Platform info: \n" <<"  " <<  pf.info() << std::endl;

        vector<viennacl::ocl::device> devices = pf.devices(CL_DEVICE_TYPE_GPU);
        num_devices[i] = devices.size();
        device_ids[i] = (cl_device_id*)malloc(num_devices[i] * sizeof(cl_device_id));

        for (size_t j = 0; j < num_devices[i]; ++j)
        {
            std::cout << "  Devices info: " <<endl << devices[j].info() << std::endl;
            device_ids[i][j] = devices[j].id();
        }

        cl_context_properties properites[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)pf.id(), 0};
        contexts[i] = clCreateContext(properites, num_devices[i], device_ids[i], NULL, NULL, &err);
        VIENNACL_ERR_CHECK(err);

        cmd_queues[i] = (cl_command_queue *)malloc(sizeof(cl_command_queue) * num_devices[i]);
        mem_matrixs[i] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_size * sizeof(ScalarType), &(square_matrix(0,0)), &err);
        VIENNACL_ERR_CHECK(err);

        for (size_t j = 0; j < num_devices[i]; ++j)
        {
            cmd_queues[i][j] = clCreateCommandQueue(contexts[i], devices[j].id(), CL_QUEUE_PROFILING_ENABLE, &err);
            if (err != CL_SUCCESS)
            {
                printf("Failed to create cmd queue for device_%d \n" ,  j);
                continue;
            }
        }
    }

    #ifdef PRINTF
        printf("Matrix input:\n");
        for (int i = 0; i < lu_dim; i++)
        {
            for (int j = 0; j < lu_dim; j++)
            {   
                printf("%.2f\t",square_matrix(i,j));
            }
            printf("\n");
        }
        printf("\n");
    #endif

    std::cout << "Matrix size: " <<  lu_dim << " * " <<  lu_dim << std::endl;
    std::cout << "----- LU factorization -----" << std::endl;

    for(int i = 0; i< num_platforms;i++)
    {
        time0 = 0.0f, time1 = 0.0f, time2 = 0.0f, time3 = 0.0f;
        /*I is the index of contexts*/
        viennacl::ocl::setup_context(i, contexts[i], *device_ids[i], *cmd_queues[i]);
        viennacl::ocl::switch_context(i); //activate the new context (only mandatory with context-id not equal to zero)

        std::cout << "Existing context: " << contexts[i] << std::endl;
        std::cout << "ViennaCL uses context: " << viennacl::ocl::current_context().handle().get() << std::endl;

        /*GPU memory*/
        viennacl::matrix<ScalarType> vcl_square_matrix(mem_matrixs[i], lu_dim, lu_dim);

        /*A = LU using OpenCL*/
        timer.start();
        viennacl::linalg::lu_factorize(vcl_square_matrix);
        time0 += timer.get();
        lu_time += time0;
        inv_time += time0 ;
        std::cout<<"LU分解GPU耗时:"<<time0 * 1e03<<" ms"<<std::endl;

        /*A = LU using boost C++*/
        timer.start();
        boost::numeric::ublas::lu_factorize(square_matrix_cpu);
        time2 += timer.get();
        std::cout<<"LU分解CPU耗时:"<<time2 * 1e03<<" ms"<<std::endl;
        /*GPU -> CPU*/
        viennacl::copy(vcl_square_matrix, square_matrix);

        float e = 0;
        for (int i = 0; i < lu_dim; i++)
        {
            for (int j = 0; j < lu_dim; j++)
            {
                e = abs(square_matrix(i,j)-square_matrix_cpu(i,j)) / (square_matrix_cpu(i,j) + 1e-09);
                if (e > 1e-05)
                {
                    printf("lu_factorize verify fail!please check it!\n\n");
                    printf("Error result: square_matrix(%d,%d) = %f, square_matrix_cpu(%d,%d) = %f\n\n", i, j, square_matrix(i,j), i, j, square_matrix_cpu(i,j));
                    return -1;
                }
            }
        }
        printf("lu_factorize verify pass!\n\n");

        #ifdef PRINTF
            printf("matrix result:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%.2f\t",square_matrix(i,j));
                }
                printf("\n");
            }
            printf("\n");
        #endif

        //square_matrix is dividede into L and U
        boost::numeric::ublas::matrix<ScalarType> lcpu(lu_dim, lu_dim);
        boost::numeric::ublas::matrix<ScalarType> ucpu(lu_dim, lu_dim);
        lcpu = E_;

        for (size_t i = 0; i < lu_dim; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                lcpu(i,j) = square_matrix(i,j);
            }
            for (size_t j = i; j < lu_dim; j++)
            {
                ucpu(i,j) = square_matrix(i,j);
            }
        }

        #ifdef PRINTF
            printf("L:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%.2f\t",lcpu(i,j));
                }
                printf("\n");
            }
            printf("\n");

            printf("U:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%.2f\t",ucpu(i,j));
                }
                printf("\n");
            }
            printf("\n");
        #endif

        std::cout << "----- Matrix Inv -----" << std::endl;

        viennacl::matrix<ScalarType> L(lu_dim, lu_dim);
        viennacl::matrix<ScalarType> U(lu_dim, lu_dim);
        viennacl::matrix<ScalarType> E(lu_dim, lu_dim);
        viennacl::matrix<ScalarType> L_inv(lu_dim, lu_dim);
        viennacl::matrix<ScalarType> U_inv(lu_dim, lu_dim);
        viennacl::matrix<ScalarType> Matrix_inv(lu_dim, lu_dim);

        boost::numeric::ublas::matrix<ScalarType> matrix_inv(lu_dim, lu_dim);
        boost::numeric::ublas::matrix<ScalarType> matrix_inv_cpu(lu_dim, lu_dim);
        boost::numeric::ublas::matrix<ScalarType> l_inv(lu_dim, lu_dim);
        boost::numeric::ublas::matrix<ScalarType> u_inv(lu_dim, lu_dim);

        viennacl::copy(lcpu, L);
        viennacl::copy(ucpu, U);
        viennacl::copy(E_, E);

        timer.start();
        U_inv = viennacl::linalg::solve(U, E, viennacl::linalg::upper_tag());
        L_inv = viennacl::linalg::solve(L, E, viennacl::linalg::lower_tag());
        Matrix_inv = viennacl::linalg::prod(U_inv, L_inv);
        time1 += timer.get();
        inv_time += time1 ;
        std::cout<<"LU求逆GPU耗时:"<<(time0 + time1) * 1e03<<" ms"<<std::endl;

        timer.start();
        u_inv = boost::numeric::ublas::solve(ucpu,E_,boost::numeric::ublas::upper_tag());
        l_inv = boost::numeric::ublas::solve(lcpu,E_,boost::numeric::ublas::lower_tag());
        matrix_inv_cpu = boost::numeric::ublas::prod(u_inv, l_inv);
        time3 += timer.get();   
        std::cout<<"LU求逆CPU耗时:"<<(time2 + time3) * 1e03<<" ms"<<std::endl;

        viennacl::copy(Matrix_inv, matrix_inv);

        for (int i = 0; i < lu_dim; i++)
        {
            for (int j = 0; j < lu_dim; j++)
            {
                e = abs(matrix_inv_cpu(i,j)-matrix_inv(i,j)) / (matrix_inv_cpu(i,j) + 1e-09);
                if (e > 1e-05)
                {
                    printf("matrix inversion verify fail! please check it!\n\n");
                    printf("Error result: matrix_inv(%d,%d) = %f, matrix_inv_cpu(%d,%d) = %f\n\n", i, j, matrix_inv(i,j), i, j, matrix_inv_cpu(i,j));
                    return -1;
                }
            }
        }
        printf("matrix inversion verify pass!\n\n");

        float sum = 0.0;
        for (int i = 0; i < lu_dim; i++)
        {
            for (int j = 0; j < lu_dim; j++)
            {
                sum += matrix_inv(i,j);
            }
        }
        printf("sum = %.2f\n",sum);

        #ifdef PRINTF
            viennacl::copy(L_inv, lcpu);
            viennacl::copy(U_inv, ucpu);

            printf("L_inv:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%f\t",lcpu(i,j));
                }
                printf("\n");
            }
            printf("\n");

            printf("U_inv:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%f\t",ucpu(i,j));
                }
                printf("\n");
            }
            printf("\n");

            printf("Matrix_inv:\n");
            for (int i = 0; i < lu_dim; i++)
            {
                for (int j = 0; j < lu_dim; j++)
                {   
                    printf("%f\t",matrix_inv(i,j));
                }
                printf("\n");
            }
            printf("\n");
        #endif
    }

    std::cout << "All platform LU分解GPU耗时 : " << lu_time  * 1e03<<" ms"<< std::endl;
    std::cout << "All platform LU求逆GPU耗时 : " << inv_time  * 1e03<<" ms"<< std::endl;
    std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

    return EXIT_SUCCESS;
}