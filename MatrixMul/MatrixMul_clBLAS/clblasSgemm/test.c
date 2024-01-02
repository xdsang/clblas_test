/**********************************************************************************************
* @Copyright    : 2023 by Innosilicon Technologies Ltd. All Rights Reserved
* @FileName     : test.c
* @Date         : May.2023
* @Author       : wusk
* @Description  : Matrix-matrix product of general square matrices with float-complex elements.
***********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include "clBLAS.h"
#include "cblas.h"
#include <sys/time.h>

static double get_wall_time()
{
    struct  timeval time;
    if (gettimeofday(&time,NULL))
        return 0;
    return (double)time.tv_sec + (double)time.tv_usec*0.000001;
}

size_t N = 16;
cl_bool PrintMatrix = 0;
unsigned int batch = 1;

void InitialMatrix(float *Mat, int rows, int cols, int batch)
{
	for (int i = 0; i < rows*cols*batch; i++)
	{
        Mat[i] = 1.0;
	}
}

void ShowMatrix(float *Mat, int lda, int rows, int cols, int batch)
{
    for (int i=0; i < rows; i++)
    {
        for (int j = 0; j < cols*batch; j++)
        {
            printf("%f ", Mat[i+j*lda]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_help(const char* name, int ret)
{
    printf("Usage: [-n matrix size] [-b batch num] [-p print on/off(bool value)]\n");
    printf("-n matrix size: user defined, n*n, default 16*16\n");
    printf("-b batch num: user defined, default 1\n");
    printf("-p print on/off(bool value): user defined, default 0, print off\n");
    exit(ret);
}

int main( int argc , char *argv[] )
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
        else if(strcmp(argv[i], "-b") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                batch = atoi(argv[i]);
            }
        }
        else if(strcmp(argv[i], "-p") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                PrintMatrix = atoi(argv[i]);
            }
        }
        else
        {
            print_help(argv[0], 1);
        }
    }

	cl_uint num_platforms, ret_num_platforms;
	if ((clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) || (num_platforms == 0))
	{
		printf( "Cannot get number of platforms!\n");
		return 1;
	}

	cl_platform_id *platform_ids = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	cl_uint *num_devices = (cl_uint *)malloc(num_platforms * sizeof(cl_uint));
	cl_device_id **device_ids = (cl_device_id **)malloc(num_platforms * sizeof(cl_device_id*));
	if (platform_ids == NULL || num_devices == NULL || device_ids == NULL)
	{
		printf("Cannot get platform_ids, not enough memory!\n");
		return 1;
	}

	if ((clGetPlatformIDs(num_platforms, platform_ids, &ret_num_platforms) != CL_SUCCESS) || (num_platforms != ret_num_platforms))
	{
		printf("Cannot get number of platforms!\n");
		return 1;
	}
	printf("Total platforms:  %d \n ",num_platforms );
	double wstart = get_wall_time();
	cl_uint total_num_devices = 0;
	cl_int errcode_ret;
	cl_int err;
    size_t i;

	for (i = 0; i < num_platforms; ++i)
	{
		char param[256];
		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(param)/sizeof(param[0]), param, NULL);
		printf("platform_%d: %s \n", i , param);

		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(param)/sizeof(param[0]), param, NULL);
		printf("Platform version: %s\n",param );

		clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &(num_devices[i]));
		if (num_devices[i] == 0)
		{
			device_ids[i] = NULL;
			printf("No gpu devices \n");
			continue;
		}

		device_ids[i] = (cl_device_id*)malloc(num_devices[i] * sizeof(cl_device_id));
		if (device_ids[i] == NULL)
		{
			num_devices[i] = 0;
			printf("No gpu devices \n");
			continue;
		}

        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, num_devices[i], device_ids[i], NULL);
        printf("Num devices of platform_%d: %d \n" ,i, num_devices[i]);
        for (int j = 0; j < num_devices[i]; ++j)
		{
			clGetDeviceInfo(device_ids[i][j], CL_DEVICE_NAME, sizeof(param)/sizeof(param[0]), param, NULL);
			printf("   Device info of device_%d: %s \n", j , param );
		}
		total_num_devices += num_devices[i];
	}

    printf("\nTotal num devices:: %d \n" , total_num_devices);
	printf("\nStart init ...\n");

	cl_context *contexts = (cl_context *)malloc(sizeof(cl_context) * num_platforms);
    cl_mem **bufA = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
    cl_mem **bufB = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
    cl_mem **bufC = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
	cl_command_queue **cmd_queues = (cl_command_queue **)malloc(sizeof(cl_command_queue* ) * num_platforms);

	if (!contexts || !bufA || !bufB || !bufC || !cmd_queues)
	{
        printf("\nInitialize object for platform error \n");
		return 1;
	}

	for (int i = 0; i < num_platforms; ++i)
	{
		contexts[i] = 0;
		bufA[i] = NULL;
		bufB[i] = NULL;
		bufC[i] = NULL;
		cmd_queues[i] =  NULL;
	}

    int ret = 0;
    double ktime = 0.0;
    /* Setup parameter */
    float alpha, beta;
    alpha = 1.0f;
    beta = 0.0f;
    /* Allocate Host Memory*/
    int row = N;
    int col = N;
    int const A_ROW = row;
    int const A_COL = col;
    int const B_ROW = row;
    int const B_COL = col;

    int const A_size = A_ROW * A_COL;
    int const B_size = B_ROW * B_COL;
    int const C_size = A_ROW * B_COL;

    float *A, *B, *C;
    A = (float *)malloc (A_size * batch * sizeof (*A));
    B = (float *)malloc (B_size * batch * sizeof (*B));
    C = (float *)malloc (C_size * batch * sizeof (*C));
    if (!A || !B || !C)
    {
        printf ("host memory allocation failed \n");
        return 1;
    }

    /* Initial Host Matrix A && B*/
    InitialMatrix(A, A_ROW, A_COL, batch);
    InitialMatrix(B, B_ROW, B_COL, batch);
    size_t lda = A_ROW;
    size_t ldb = B_ROW;
    size_t ldc = A_ROW;

    if (PrintMatrix == 1)
    {
        printf("\nA: \n");
        ShowMatrix(A, lda, A_ROW, A_COL, batch);
        printf("\nB: \n");
        ShowMatrix(B, ldb, B_ROW, B_COL, batch);
    }

	for (i = 0; i < num_platforms; ++i)
	{
		printf("Init platform_%d \n" , i);
		cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids[i]), 0};
		if (num_devices[i] == 0)
		{
			printf("No device\n");
			continue;
		}
		bufA[i] = (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufB[i] = (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufC[i] = (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
		contexts[i] = clCreateContext(prop, num_devices[i], device_ids[i], NULL, NULL, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			printf("Create context failed\n");
			continue;
		}

		cmd_queues[i] = (cl_command_queue *)malloc(sizeof(cl_command_queue) * num_devices[i]);
		if (cmd_queues[i] == NULL)
		{
			printf("Failed to allocate memory for cmd queues \n");
			continue;
		}

		for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
		{
            bufA[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, A_size * sizeof(*A),
                                 NULL, &err);
            bufB[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, B_size * sizeof(*B),
                                 NULL, &err);
            bufC[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, C_size * sizeof(*C),
                                 NULL, &err);
			cmd_queues[i][dev_id] = clCreateCommandQueue(contexts[i], device_ids[i][dev_id], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
			if (errcode_ret != CL_SUCCESS)
			{
				printf("Failed to create cmd queue for device_%d \n" , dev_id);
				continue;
			}
		}
    }

    /* Setup clBLAS */
    err = clblasSetup( );
    if (err != CL_SUCCESS)
    {
        printf("clblasSetup() failed with %d\n", err);
        for (int i = 0; i < num_platforms; ++i)
        {
            clReleaseContext(contexts[i]);
            for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
            {
                if (cmd_queues[i][dev_id] != NULL)
                {
                    clReleaseCommandQueue(cmd_queues[i][dev_id]);
                }
            }
        }
        return 1;
    }

    for (size_t j = 0; j < batch; j++)
    {
        for (i = 0; i < num_platforms; ++i)
        {
            for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
            {
                /*Place Matrices inside GPU Memory*/
                err = clEnqueueWriteBuffer(cmd_queues[i][dev_id], bufA[i][dev_id], CL_TRUE, 0,
                                           A_size * sizeof(*A), A + j * A_size, 0, NULL, NULL);
                err = clEnqueueWriteBuffer(cmd_queues[i][dev_id], bufB[i][dev_id], CL_TRUE, 0,
                                           B_size * sizeof(*B), B + j * B_size, 0, NULL, NULL);
				clFinish(cmd_queues[i][dev_id]);
            }
        }
    }

    double parallel_start = get_wall_time()* 1000.0f;
    for (size_t j = 0; j < batch; j++)
    {
        for (i = 0; i < num_platforms; ++i)
        {
            for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
            {
                double kstart = get_wall_time();
                /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
                err = clblasSgemm( clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                                    A_ROW, B_COL, A_COL,
                                    alpha, bufA[i][dev_id], 0, lda,
                                    bufB[i][dev_id], 0, ldb, beta,
                                    bufC[i][dev_id], 0, ldc,
                                    1, &cmd_queues[i][dev_id], 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    printf("clblasCgemm() failed with %d\n", err);
                    ret = 1;
                }
            }
        }
    }

    for (size_t j = 0; j < batch; j++)
    {
        for (i = 0; i < num_platforms; ++i)
        {
            for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
            {
                if (cmd_queues[i][dev_id] != NULL)
                {
                    clFinish(cmd_queues[i][dev_id]);
                }
            }
        }
    }

    double parallel_end = get_wall_time()* 1000.0f;
    double parallel_runTime = ( parallel_end - parallel_start) / (batch * 1.0f);
    printf("\nParallel start time is:  %0.6f , parallel end time is:  %0.6f , parallel running time %0.6f (ms)\n ",parallel_start, parallel_end, parallel_runTime );
    for (size_t j = 0; j < batch; j++)
    {
        for (i = 0; i < num_platforms; ++i)
        {
            for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
            {
                 /* Fetch results of calculations from GPU memory. */
                err = clEnqueueReadBuffer(cmd_queues[i][dev_id], bufC[i][dev_id], CL_TRUE, 0,
                                          C_size * sizeof(*C),
                                          C + j * C_size, 0, NULL, NULL);
				clFinish(cmd_queues[i][dev_id]);
            }
        }
    }

    double wend = get_wall_time();
    double runTime = ((wend - wstart) / (batch*1.0f)) * 1000.0f;

    if (PrintMatrix == 1)
    {
        printf("\nAxB: \n");
        ShowMatrix(C, ldc, A_ROW, B_COL, batch);
    }

    float* C_result = (float *)malloc (C_size * batch * sizeof (*C_result));
    for (size_t i = 0; i < batch; i++)
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A_ROW, B_COL, A_COL, alpha,
                A + i*A_size, A_ROW, B + i*B_size, B_ROW, beta, C_result + i*C_size, A_ROW);
    }

    float e = 0;
    for (size_t i = 0; i < C_size*batch; i++)
    {
        e = abs(C_result[i] - C[i])/(C_result[i] + 1e-09);
        if (e > 1e-05)
        {
            printf("MatrixMul verify fail!please check it!\n\n");
            printf("Error result: C[%d] = %f, C_result[%d] = %f\n\n", i, C[i], i, C_result[i]);
            return -1;
        }
    }
    printf("\nMatrixMul verify pass! \n");
    printf("\nOpenCL execution time is:%0.3f ms\n", runTime);

    /* Release OpenCL events. */
	for (i = 0; i < num_platforms; ++i)
	{
		/* Release OpenCL working objects. */
		clReleaseContext(contexts[i]);
		for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
		{
            /* Release OpenCL memory objects. */
            clReleaseMemObject(bufA[i][dev_id]);
            clReleaseMemObject(bufB[i][dev_id]);
            clReleaseMemObject(bufC[i][dev_id]);
            clReleaseDevice(device_ids[i][dev_id]);
			clReleaseCommandQueue(cmd_queues[i][dev_id]);
		}
	}

    /* Finalize work with clBLAS */
    clblasTeardown( );

    return ret;
}
