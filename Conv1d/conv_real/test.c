
/*******************************************************************************
* @Copyright    : 2023 by Innosilicon Technologies Ltd. All Rights Reserved
* @Description  : conv1d of vector with float-complex elements
********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "CL/cl.h"
#include <thread>

#include <string.h>
#ifdef WINDOWS
#include <time.h>
#include <windows.h>
LARGE_INTEGER cpuFreq;
LARGE_INTEGER startTime;
LARGE_INTEGER endTime;
#else
#include <sys/time.h>
static double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
        return 0;
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}
#endif

size_t N = 128;
size_t shift = 7;
size_t kSize = 8;
size_t LOOPNUM = 1;
cl_bool PrintVector = 0;

void InitialVector(float *input, int size)
{
    for (int i = 0; i < size; i++)
    {
        input[i] = 1.0;
    }
}

void ShowVector(float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", output[i]);
    }
    printf("\n");
}

static void print_help(const char *name, int ret)
{
    printf("Usage: [-n vector size] [-k filter size] [-l loop num] [-p print or not(bool value)]\n");
    printf("-n shift: user defined, default value 7, which means 1<<7, i.e, vector size = 2^7 = 128\n");
    printf("-k filter size: user defined, not more than 1024, default 8\n");
    printf("-l loop num: user defined, default 1\n");
    printf("-p print on/off(bool value): user defined, default 0, print off\n");
    exit(ret);
}

char *loadFile(const char *file)
{
    FILE *program_handle;
    size_t program_size;
    char *program_buffer;
    program_handle = fopen(file, "r");

    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_buffer;
}

int main(int argc, char *argv[])
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
                shift = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "-k") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                kSize = atoi(argv[i]);
                if (kSize > 1024)
                {
                    printf("Filter size can't be greater than 1024! Retry it!\n");
                    return 1;
                }
            }
        }
        else if (strcmp(argv[i], "-l") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                LOOPNUM = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            i++;
            if (i == argc)
            {
                print_help(argv[0], 1);
            }
            else
            {
                PrintVector = atoi(argv[i]);
            }
        }
        else
        {
            print_help(argv[0], 1);
        }
    }

    N = 1 << shift;
    cl_uint num_platforms, ret_num_platforms;

    if ((clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) || (num_platforms == 0))
    {
        printf("Cannot get number of platforms!\n");
        return 1;
    }

    cl_platform_id *platform_ids = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    cl_uint *num_devices = (cl_uint *)malloc(num_platforms * sizeof(cl_uint));
    /*Malloc devices for every platform*/
    cl_device_id **device_ids = (cl_device_id **)malloc(num_platforms * sizeof(cl_device_id *));

    if (platform_ids == NULL || num_devices == NULL || device_ids == NULL)
    {
        printf("Cannot get platform_ids, not enough memory!\n");
        return 1;
    }
    /*Get platforms numbers*/
    if ((clGetPlatformIDs(num_platforms, platform_ids, &ret_num_platforms) != CL_SUCCESS) || (num_platforms != ret_num_platforms))
    {
        printf("Cannot get number of platforms!\n");
        return 1;
    }

    printf("Total CL platforms: %d\n", num_platforms);

    cl_uint total_num_devices = 0;
    char *fileName = "conv1d.cl";
    const char *source = loadFile("conv1d.cl");
    int i;
    cl_int errcode_ret;
    int ret = 0;
    cl_int err;

    std::thread tps[num_platforms];
    /*Get devices for every platform */
    for (i = 0; i < num_platforms; ++i)
    {
        char param[256];

        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(param) / sizeof(param[0]), param, NULL);
        printf("Platform_%d : %s \n", i, param);

        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(param) / sizeof(param[0]), param, NULL);
        printf("Platform version %s\n", param);

        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &(num_devices[i]));
        if (num_devices[i] == 0)
        {
            device_ids[i] = NULL;
            printf("No gpu devices \n");
            continue;
        }

        device_ids[i] = (cl_device_id *)malloc(num_devices[i] * sizeof(cl_device_id));
        if (device_ids[i] == NULL)
        {
            num_devices[i] = 0;
            printf("No gpu devices \n");
            continue;
        }

        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, num_devices[i], device_ids[i], NULL);

        printf("Num devices of platform_%d: %d \n", i, num_devices[i]);
        for (int j = 0; j < num_devices[i]; ++j)
        {
            clGetDeviceInfo(device_ids[i][j], CL_DEVICE_NAME, sizeof(param) / sizeof(param[0]), param, NULL);
            printf("Device info of device_%d: %s \n", j, param);
        }
        total_num_devices += num_devices[i];
    }

    printf("\nTotal num devices:%d \n", total_num_devices);
    printf("Start init ...\n");

    /*Init context ,program, kernel,buffer,commandqueue for platforms*/
    cl_context *contexts = (cl_context *)malloc(sizeof(cl_context) * num_platforms);
    cl_program *programs_for_kernel = (cl_program *)malloc(sizeof(cl_program) * num_platforms);
    cl_kernel *crypt_kernels = (cl_kernel *)malloc(sizeof(cl_kernel) * num_platforms);
    cl_mem **bufA = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
    cl_mem **bufH = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
    cl_mem **bufRes = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
    /*Malloc each command queue for each device for each platform*/
    cl_command_queue **cmd_queues = (cl_command_queue **)malloc(sizeof(cl_command_queue *) * num_platforms);

    if (!contexts || !programs_for_kernel || !crypt_kernels || !bufA || !bufH || !bufRes || !cmd_queues)
    {
        printf("\nInitialize object for platform error \n");
        return 1;
    }

    for (i = 0; i < num_platforms; ++i)
    {
        contexts[i] = 0;
        programs_for_kernel[i] = 0;
        crypt_kernels[i] = 0;
        bufA[i] = NULL;
        bufH[i] = NULL;
        bufRes[i] = NULL;
        cmd_queues[i] = NULL;
    }

    /*Malloc host memory*/
    float *A, *H, *Res;
    A = (float *)malloc(N * sizeof(*A));
    H = (float *)malloc(kSize * sizeof(*H));
    Res = (float *)malloc((N + kSize - 1) * sizeof(*Res));

    if (!A || !H || !Res)
    {
        printf("Host memory allocation failed \n");
        return 1;
    }

    /* Initial Host Vector*/
    InitialVector(A, N);
    InitialVector(H, kSize);

    if (PrintVector == 1)
    {
        printf("\nVector: \n");
        ShowVector(A, N);
        printf("\nFilter: \n");
        ShowVector(H, kSize);
    }

    double ktime = 0.0;
    /* Work Space*/
    size_t local_work_size = (kSize + 128 - 1) / 128 * 128;
    size_t global_work_size = ((N + kSize - 1) + local_work_size - 1) / local_work_size * local_work_size;
    size_t localBufSize = kSize + local_work_size - 1;

    /* Initialize for each platform*/
    for (i = 0; i < num_platforms; ++i)
    {
        int inuse_devices = 0;

        cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids[i]), 0};
        if (num_devices[i] == 0)
        {
            printf("No device\n");
            continue;
        }

        contexts[i] = clCreateContext(prop, num_devices[i], device_ids[i], NULL, NULL, &errcode_ret);
        if (errcode_ret != CL_SUCCESS)
        {
            printf("Create context failed\n");
            continue;
        }

        /* Prepare OpenCL Memory objects */
        bufA[i] = (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufH[i] =  (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufRes[i] =  (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);

        /*Create command queue for each device*/
        cmd_queues[i] = (cl_command_queue *)malloc(sizeof(cl_command_queue) * num_devices[i]);
        if (cmd_queues[i] == NULL)
        {
            printf("Failed to allocate memory for cmd queues \n");
            continue;
        }

        for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
        {
            cmd_queues[i][dev_id] = clCreateCommandQueue(contexts[i], device_ids[i][dev_id], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
            if (errcode_ret != CL_SUCCESS)
            {
                printf("Failed to create cmd queue for device_%d \n", dev_id);
                continue;
            }
            bufA[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, N * sizeof(*A),
                                    NULL, &err);
            bufH[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, kSize * sizeof(*H),
                                    NULL, &err);
            bufRes[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, (N + kSize - 1) * sizeof(*Res),
                                    NULL, &err);
            inuse_devices += 1;
        }

        programs_for_kernel[i] = clCreateProgramWithSource(contexts[i], 1, &source, NULL, &err);
        if (err)
            printf("CLcreate program with source error: %d\n", err);

        /*Build kernels for deivces*/
        if (clBuildProgram(programs_for_kernel[i], num_devices[i], device_ids[i], NULL, NULL, NULL) != CL_SUCCESS)
        {
            size_t len = 0;
            if (clGetProgramBuildInfo(programs_for_kernel[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len) == CL_SUCCESS)
            {
                char *buffer = (char *)calloc(len, sizeof(char));
                if (clGetProgramBuildInfo(programs_for_kernel[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL) == CL_SUCCESS)
                {
                    printf("Program build info %s \n", buffer);
                }
            }
            continue;
        }

        crypt_kernels[i] = clCreateKernel(programs_for_kernel[i], "conv", &errcode_ret);
        if (errcode_ret != CL_SUCCESS)
        {
            printf("Create crypt kernel failed \n");
            continue;
        }
    }
#ifdef WINDOWS
    QueryPerformanceFrequency(&cpuFreq);
    QueryPerformanceCounter(&startTime);
#else
    double wstart = get_wall_time()* 1000.0f;
#endif

    for (int p = 0; p < num_platforms; ++p)
    {
        for (int dev_id = 0; dev_id < num_devices[p]; ++dev_id)
        {
            err = clEnqueueWriteBuffer(cmd_queues[p][dev_id], bufA[p][dev_id], CL_TRUE, 0,
                                        N * sizeof(*A), A, 0, NULL, NULL);
            err = clEnqueueWriteBuffer(cmd_queues[p][dev_id], bufH[p][dev_id], CL_TRUE, 0,
                                        kSize * sizeof(*H), H, 0, NULL, NULL);
            /* Set Parameter */
            clSetKernelArg(crypt_kernels[p], 0, sizeof(cl_mem), (void *)&bufA[p][dev_id]);
            clSetKernelArg(crypt_kernels[p], 1, sizeof(cl_int), (void *)&N);
            clSetKernelArg(crypt_kernels[p], 2, sizeof(cl_mem), (void *)&bufH[p][dev_id]);
            clSetKernelArg(crypt_kernels[p], 3, sizeof(cl_int), (void *)&kSize);
            clSetKernelArg(crypt_kernels[p], 4, sizeof(cl_mem), (void *)&bufRes[p][dev_id]);
            clSetKernelArg(crypt_kernels[p], 5, localBufSize * sizeof(cl_float2), NULL);
            clFinish(cmd_queues[p][dev_id]);
        }
    }

    double parallel_start = get_wall_time()* 1000.0f;

    for (int p = 0; p < num_platforms ; ++p)
    {
        for (int dev_id = 0; dev_id < num_devices[p]; ++dev_id)
        {
            err = clEnqueueNDRangeKernel(cmd_queues[p][dev_id], crypt_kernels[p], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
            clFlush(cmd_queues[p][dev_id]);
        }
    }

    double parallel_middle = get_wall_time()* 1000.0f;

    for (int p = 0; p < num_platforms ; ++p)
    {
        for (int dev_id = 0; dev_id < num_devices[p]; ++dev_id)
        {
            if (cmd_queues[p][dev_id] != NULL)
            {
                err = clFinish(cmd_queues[p][dev_id]);
                if (err != CL_SUCCESS)
                {
                    return 1;
                }
            }
        }
    }

    double parallel_end = get_wall_time()* 1000.0f;
    double parallel_runTime = ( parallel_end - parallel_start);
    printf("\nParallel start time is:  %0.6f  , parallel end time is:  %0.6f , parallel running time %0.6f (ms)\n ",parallel_start,  parallel_end, parallel_runTime );
    printf("\nOCL enqueue NDRange kernel time is:  %0.6f   (ms)\n ",( parallel_middle - parallel_start) );

    for (i = 0; i < num_platforms; ++i)
    {
        for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
        {
            /* Fetch results of calculations from GPU memory. */
            err = clEnqueueReadBuffer(cmd_queues[i][dev_id], bufRes[i][dev_id], CL_TRUE, 0,
                                        (N + kSize - 1) * sizeof(*Res),
                                        Res, 0, NULL, NULL);
            clFinish(cmd_queues[i][dev_id]);
        }
    }

#ifdef WINDOWS
    QueryPerformanceCounter(&endTime);
    double runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart) / (iteration * 1.0f);
#else
    double wend = get_wall_time()* 1000.0f;
    double runTime = (wend - wstart) ;
#endif
    printf("\nOpenCL execution time is: %0.6f ms\n", runTime);
    printf("\nVector size: %d, filter size: %d, loop times: %d  \n", N, kSize, LOOPNUM);

    /**************************计算误差**************************/
    float *A_ = (float *)malloc((N + kSize - 1) * sizeof(*A_));
    memset(A_, 0, (N + kSize - 1) * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        A_[i] = A[i];
    }

    float *Res_cpu = (float *)malloc((N + kSize - 1) * sizeof(*Res_cpu));
    for (int i = 0; i < N + kSize - 1; i++)
    {
        float temp = 0.0;
        for (int j = 0; j < kSize; j++)
        {
            if (i >= j)
            {
                temp += H[j] * A_[i - j];
            }
        }
        Res_cpu[i] = temp;
    }

    float e = 0;
    for (size_t i = 0; i < N + kSize - 1; i++)
    {
        e = abs(Res_cpu[i] - Res[i]) / (Res_cpu[i] + 1e-09);
        if (e > 1e-05)
        {
            printf("Conv1d verify fail!please check it!\n\n");
            printf("Error result: Res[%d] = %f, Res_cpu[%d] = %f\n\n", i, Res[i], i, Res_cpu[i]);
            return -1;
        }
    }
    printf("\nConv1d verify pass! \n");
    printf("\nGroup_num: %d, group_size: %d \n", global_work_size / local_work_size, local_work_size);
    printf("\n");

    if (PrintVector == 1)
    {
        printf("\nResult: \n");
        ShowVector(Res, (N + kSize - 1));
    }

    /* Release Host memory objects */
    if (A)
        free(A);
    if (H)
        free(H);
    if (Res)
        free(Res);

    //release ocl object
    for (i = 0; i < num_platforms; ++i)
    {
        /* Release OpenCL working objects. */
        clReleaseProgram(programs_for_kernel[i]);
        clReleaseContext(contexts[i]);
        clReleaseKernel(crypt_kernels[i]);

        for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
        {
            clReleaseMemObject(bufA[i][dev_id]);
            clReleaseMemObject(bufH[i][dev_id]);
            clReleaseMemObject(bufRes[i][dev_id]);
            clReleaseCommandQueue(cmd_queues[i][dev_id]);
        }
    }

    return ret;
}
