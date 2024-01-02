#define _CRT_SECURE_NO_WARNINGS
//#define WINDOWS
//#define double_queue //默认不开
#include <iostream>
#include <string>
#include "CL/cl.h"
#include <sys/types.h>
#include <stdio.h>
#include <fstream>

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
	struct  timeval time;
	if (gettimeofday(&time, NULL))
		return 0;
	return (double)time.tv_sec + (double)time.tv_usec*0.000001;
}
#endif
using namespace std;

char  *ReadKernelFile(const char *fileName, size_t *length)
{
	FILE *file = NULL;
	size_t sourceLength;
	char *sourceString;
	int ret;

	file = fopen(fileName, "rb");
	if (file == NULL)
	{
		printf("Can't open %s\n", fileName);
		return NULL;
	}

	fseek(file, 0, SEEK_END);
	sourceLength = ftell(file);
	fseek(file, 0, SEEK_SET);
	sourceString = (char*)malloc(sourceLength + 1);
	sourceString[0] = '\0';

	ret = fread(sourceString, sourceLength, 1, file);
	if (ret == 0)
	{
		printf("Can't read source %s\n", fileName);
		return NULL;
	}
	fclose(file);
	if (length != 0)
	{
		*length = sourceLength;
	}
	sourceString[sourceLength] = '\0';

	return sourceString;
}

#ifdef WINDOWS
int main()
{
	int iteration=1;
#else
int main(int argc , char *argv[])
{
	if(argc < 3)
	{
		printf("Please check your input loop num and batchsize,for example: ./fft1d 1 40(loop=1,batchsize=40) \n exit now!");
		return -1;
    }

	int iteration=atoi(argv[1]);
	int batchsize=atoi(argv[2]);
#endif
	cl_uint num_platforms, ret_num_platforms;

	if ((clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) || (num_platforms == 0))
	{
		cout << "Cannot get number of platforms!" << endl;
		return 1;
	}
	
	cl_platform_id *platform_ids = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	cl_uint *num_devices = (cl_uint *)malloc(num_platforms * sizeof(cl_uint));
	cl_device_id **device_ids = (cl_device_id **)malloc(num_platforms * sizeof(cl_device_id*));

	if (platform_ids == NULL || num_devices == NULL || device_ids == NULL)
	{
		cout << "Cannot get platform_ids, not enough memory!" << endl;
		return 1;
	}
	if ((clGetPlatformIDs(num_platforms, platform_ids, &ret_num_platforms) != CL_SUCCESS) || (num_platforms != ret_num_platforms))
	{
		cout << "Cannot get number of platforms!" << endl;
		return 1;
	}
	
	cout << "Total platforms:  " << num_platforms << endl;
	
	cl_uint total_num_devices = 0;


	size_t program_length;
	char *fileName = "kernel.cl";
	const char *source = ReadKernelFile("kernel.cl", &program_length);

	char *fileName2 = "kernel2.cl";
	size_t program_length2;
	const char *source2 = ReadKernelFile("kernel2.cl", &program_length2);

	int i;
	cl_int errcode_ret;
	int ret = 0;
	cl_int err;
    double ktime = 0.0;

	for (i = 0; i < num_platforms; ++i)
	{
		char param[256];

		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(param)/sizeof(param[0]), param, NULL);
		cout << " -> platform_" << i << " : " << param << " ";
		
		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(param)/sizeof(param[0]), param, NULL);
		cout << param << " " << endl;

		clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &(num_devices[i]));
		if (num_devices[i] == 0)
		{
			device_ids[i] = NULL;

			cout << " (no gpu devices) ";
			cout << endl;
			continue;
		}

		device_ids[i] = (cl_device_id*)malloc(num_devices[i] * sizeof(cl_device_id));
		if (device_ids[i] == NULL)
		{
			num_devices[i] = 0;

			cout << " (no enough memory) ";
			cout << endl;
			continue;
		}
		
		clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, num_devices[i], device_ids[i], NULL);
		for (int j = 0; j < num_devices[i]; ++j)
		{
			clGetDeviceInfo(device_ids[i][j], CL_DEVICE_NAME, sizeof(param)/sizeof(param[0]), param, NULL);
			cout << "	 -> device_"<< j << " : "<< param << " " << endl;
		}
		cout << endl;
		total_num_devices += num_devices[i];
	}

    cout << "\nTotal num devices " << total_num_devices << endl;

	int dataSize = 64 * 1024 * batchsize;
	cl_context *contexts = (cl_context *)malloc(sizeof(cl_context) * num_platforms);
	cl_program *programs_for_kernel_one = (cl_program *)malloc(sizeof(cl_program) * num_platforms);
	cl_program *programs_for_kernel_two = (cl_program *)malloc(sizeof(cl_program) * num_platforms);
	cl_kernel *crypt_kernels_one = (cl_kernel *)malloc(sizeof(cl_kernel) * num_platforms);
	cl_kernel *crypt_kernels_two = (cl_kernel *)malloc(sizeof(cl_kernel) * num_platforms);
	cl_mem **bufA1 = (cl_mem **)malloc(sizeof(cl_mem*) * num_platforms);
	cl_mem **bufB1 = (cl_mem  **)malloc(sizeof(cl_mem*) * num_platforms);
	cl_mem **bufC1 = (cl_mem  **)malloc(sizeof(cl_mem*) * num_platforms);
	cl_command_queue **cmd_queues = (cl_command_queue **)malloc(sizeof(cl_command_queue* ) * num_platforms);
	cl_event **events = (cl_event **)malloc(sizeof(cl_event* ) * num_platforms);

	if (!contexts || !programs_for_kernel_one || !programs_for_kernel_two|| !crypt_kernels_one || !crypt_kernels_two || !bufA1 || !bufB1 || !bufC1 || !cmd_queues)
	{
        cout << "\nInitial object for platform error " << endl;
		return 1;
	}
	
	for (i = 0; i < num_platforms; ++i)
	{
		contexts[i] = 0;
		programs_for_kernel_one[i] = 0;
		programs_for_kernel_two[i] = 0;
		crypt_kernels_one[i] = 0;
		crypt_kernels_two[i] = 0;
		bufA1[i] = NULL;
		bufB1[i] = NULL;
		bufC1[i] = NULL;
		cmd_queues[i] =  NULL;
		events[i] =  NULL;		
	}
	
	for (i = 0; i < num_platforms; ++i)
	{
		int inuse_devices = 0;

		cout << "  -> init platform_" << i << endl;
		cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids[i]), 0};
		if (num_devices[i] == 0)
		{
			cout << "    -> no device" << endl;
			continue;
		}

		contexts[i] = clCreateContext(prop, num_devices[i], device_ids[i], NULL, NULL, &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			cout << "    -> create context failed" << endl;
			continue;
		}

		cmd_queues[i] = (cl_command_queue *)malloc(sizeof(cl_command_queue) * num_devices[i]);
		if (cmd_queues[i] == NULL)
		{
			cout << "    -> failed to allocate memory for cmd queues " << endl;
			continue;
		}

		events[i] = (cl_event *)malloc(sizeof(cl_event) * num_devices[i]);
		if (events[i] == NULL)
		{
			printf("Failed to allocate memory for cmd queues\n ");
			continue;
		}
        /* Prepare OpenCL Memory objects */
        bufA1[i] = (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufB1[i] =  (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);
        bufC1[i] =  (cl_mem *)malloc(sizeof(cl_mem) * num_devices[i]);

		for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
		{
			cmd_queues[i][dev_id] = clCreateCommandQueue(contexts[i], device_ids[i][dev_id], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
			if (errcode_ret != CL_SUCCESS)
			{
				cout << "    -> failed to create cmd queue for device_" << dev_id << endl;
				continue;
			}
			bufA1[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, dataSize * 2 * sizeof(float),NULL, &err);
			bufB1[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_ONLY, dataSize  * sizeof(float),NULL, &err);
			bufC1[i][dev_id] = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE, dataSize * 2 * sizeof(float),NULL, &err);
			inuse_devices += 1;
		}

		programs_for_kernel_one[i] = clCreateProgramWithSource(contexts[i], 1, &source, NULL, &err);
		if (err)
			printf("clCreateProgramWithSource error: %d\n", err);

		programs_for_kernel_two[i] = clCreateProgramWithSource(contexts[i], 1, &source2, NULL, &err);
		if (err)
			printf("clCreateProgramWithSource error: %d\n", err);

		if (clBuildProgram(programs_for_kernel_one[i], num_devices[i], device_ids[i], NULL, NULL, NULL) != CL_SUCCESS)
		{
			cout << "    -> build program failed for all devices" << endl;

			size_t len = 0;
			if (clGetProgramBuildInfo(programs_for_kernel_one[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len) == CL_SUCCESS)
			{
				char *buffer = (char *)calloc(len, sizeof(char));
				if (clGetProgramBuildInfo(programs_for_kernel_one[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL) == CL_SUCCESS)
				{
					cout << buffer << endl;
				}
			}
			continue;
		}

		if (clBuildProgram(programs_for_kernel_two[i], num_devices[i], device_ids[i], NULL, NULL, NULL) != CL_SUCCESS)
		{
			cout << "    -> build program failed for all devices" << endl;
			size_t len = 0;

			if (clGetProgramBuildInfo(programs_for_kernel_two[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len) == CL_SUCCESS)
			{
				char *buffer = (char *)calloc(len, sizeof(char));
				if (clGetProgramBuildInfo(programs_for_kernel_two[i], device_ids[i][0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL) == CL_SUCCESS)
				{
					cout << buffer << endl;
				}
			}
			continue;
		}

		crypt_kernels_one[i] = clCreateKernel(programs_for_kernel_one[i], "fft_fwd", &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			cout << "    -> create crypt kernel failed" << endl;
			continue;
		}

		crypt_kernels_two[i] = clCreateKernel(programs_for_kernel_two[i], "fft_fwd", &errcode_ret);
		if (errcode_ret != CL_SUCCESS)
		{
			cout << "    -> create crypt kernel failed" << endl;
			continue;
		}
	}

	float *A1, *B1, *C1;
	A1 = (float *)malloc(dataSize * 2 * sizeof(float));
	B1 = (float *)malloc(dataSize * sizeof(float));
	C1 = (float *)malloc(dataSize * 2 * sizeof(float));

    if (!A1 || !B1 || !C1)
	{
		printf("host memory allocation failed \n");
		return EXIT_FAILURE;
    }

	size_t global_work_size_kernel1[1] = { 256 * 1 *batchsize }; 
	size_t local_work_size_kernel1[1] = { 256 };

	size_t global_work_size_kernel2[1] = { 256 * 1*batchsize };
	size_t local_work_size_kernel2[1] = { 256 };    

	for(int j=0;j<batchsize;j++)
	{
		for (int i = 0; i<64*1024; i++)
		{
			B1[64*1024*j+i] = i;
		}
	}
    #ifdef WINDOWS
        QueryPerformanceFrequency(&cpuFreq);
        QueryPerformanceCounter(&startTime);
    #else
        double wstart = get_wall_time();
    #endif

    //loop
    for (int j = 0; j<iteration; j++)
    {
		for (i = 0; i < num_platforms; ++i)
		{
			for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
			{
				err = clEnqueueWriteBuffer(cmd_queues[i][dev_id], bufB1[i][dev_id], CL_FALSE, 0, dataSize  * sizeof(float), B1, 0, NULL, NULL);
				clSetKernelArg(crypt_kernels_one[i], 0, sizeof(cl_mem), &bufB1[i][dev_id]);
				clSetKernelArg(crypt_kernels_one[i], 1, sizeof(cl_mem), &bufC1[i][dev_id]);
				clSetKernelArg(crypt_kernels_two[i], 0, sizeof(cl_mem), &bufC1[i][dev_id]);
				clSetKernelArg(crypt_kernels_two[i], 1, sizeof(cl_mem), &bufA1[i][dev_id]);
				clFinish(cmd_queues[i][dev_id]);
			}
		}
	}

	double parallel_start = get_wall_time()* 1000.0f;
	for (int j = 0; j<iteration; j++)
	{
		for (i = 0; i < num_platforms; ++i)
		{
			for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
			{
				err = clEnqueueNDRangeKernel(cmd_queues[i][dev_id], crypt_kernels_one[i], 1, NULL, global_work_size_kernel1, local_work_size_kernel1, 0, NULL, &events[i][dev_id]);
				err = clEnqueueNDRangeKernel(cmd_queues[i][dev_id], crypt_kernels_two[i], 1, NULL, global_work_size_kernel2, local_work_size_kernel2, 0, NULL, &events[i][dev_id]);
			}
		}
	}

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
	double parallel_end = get_wall_time()* 1000.0f;
	double parallel_runTime = ( parallel_end - parallel_start) / (iteration * 1.0f);
	printf("\nParallel start time is:  %0.6f , parallel end time is:  %0.6f , parallel running time %0.6f (ms)\n ",parallel_start, parallel_end, parallel_runTime );

	for (int j = 0; j<iteration; j++)
	{
		for (i = 0; i < num_platforms; ++i)
		{
			for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
			{
				err = clEnqueueReadBuffer(cmd_queues[i][dev_id], bufA1[i][dev_id], CL_FALSE, 0, dataSize * 2 * sizeof(float), A1, 0, NULL, NULL);
				clFinish(cmd_queues[i][dev_id]);
			}
		}
	}

#ifdef WINDOWS
	QueryPerformanceCounter(&endTime);
	double runTime = (((endTime.QuadPart - startTime.QuadPart)*1000.0f) / cpuFreq.QuadPart) / (iteration*1.0f);
#else
	double wend = get_wall_time();
	double runTime = ((wend - wstart) / (iteration*1.0f)) * 1000.0f;
#endif
	printf("\niteration : %d, batch: %d, kernel time: %f ms \n",iteration,  batchsize,  ktime*(1e-06)/(iteration*1.0f));
	printf("OpenCL execution total time is:%0.3f ms\n", runTime);

	printf("bufA1=%f,bufA1=%f,bufA1=%f,bufA1=%f,bufA1[%d]=%f,bufA1[%d]=%f\n", A1[0], A1[1], A1[2], A1[3],4096*(batchsize/2)*2, A1[4096*(batchsize/2)*2],4096*(batchsize-1)*2, A1[4096*(batchsize-1)*2]);

	for(uint i=0;i<20;i++)
	{
		printf("bufA1[%d]=%f\n",i,A1[i]);
	}
	for(uint i=4096-20;i<4095;i++)
	{
		printf("bufA1[%d]=%f\n",i,A1[i]);
	}

	/**************************计算误差**************************/
	float FFT_Result_Real[64*1024];
	float FFT_Result_Img[64*1024];
	float *FFT_err_Real,*FFT_err_Img;
	
	FFT_err_Real = (float *)malloc(dataSize * sizeof(float));
	FFT_err_Img = (float *)malloc(dataSize * sizeof(float));
	
	float max_dist_Real=0;
	float max_dist_Img=0;
	int max_dist_Real_ID=0;
	int max_dist_Img_ID=0;
	float MRE_Real=0;
	float MRE_Img=0;

	ifstream in_Real("CLFFT64K_Result_Real.txt");
	for(int i=0;i<64*1024;i++)
	{
		in_Real>>FFT_Result_Real[i];
	}

	ifstream in_Img("CLFFT64K_Result_Img.txt");
	for(int i=0;i<64*1024;i++)
	{
		in_Img>>FFT_Result_Img[i];
	}

	for(int i=0;i<dataSize;i++)
	{
		FFT_err_Real[i]=A1[i*2]-FFT_Result_Real[i%(64*1024)];
		if(max_dist_Real<FFT_err_Real[i])
		{
			max_dist_Real=FFT_err_Real[i];
			max_dist_Real_ID=i;
		}
		if(FFT_Result_Real[i%(64*1024)]!=0)
		{
			FFT_err_Real[i]=abs(FFT_err_Real[i])/abs(FFT_Result_Real[i%(64*1024)])*100;
		}

		FFT_err_Img[i]=A1[i*2+1]-FFT_Result_Img[i%(64*1024)];
		if(max_dist_Img<FFT_err_Img[i])
		{
			max_dist_Img=FFT_err_Img[i];
			max_dist_Img_ID=i;
		}

		if(FFT_Result_Img[i%(64*1024)]!=0)
		{
			FFT_err_Img[i]=abs(FFT_err_Img[i])/abs(FFT_Result_Img[i%(64*1024)])*100;
		}		
		MRE_Real=FFT_err_Real[i]+MRE_Real;
		MRE_Img=FFT_err_Img[i]+MRE_Img;
	}
	MRE_Real=MRE_Real/dataSize;
	MRE_Img=MRE_Img/dataSize;

	if(MRE_Real<0.05 && MRE_Img<0.05)
		printf("Pass!\n");
	else
	{
		printf("偏差较大，结果也许不对\n");
		printf("平均相对误差:MRE_Real=%f% MRE_Img=%f%(若大于%0.05，认为偏差较大)\n",MRE_Real,MRE_Img);
		printf("误差最大点的实部ID %d 误差最大点的实部值 %f\n误差最大点的虚部ID %d 误差最大点的虚部值 %f\n",max_dist_Real_ID,max_dist_Real,max_dist_Img_ID,max_dist_Img);
		/******************************将最大误差的点的值和误差都打印出来******************************/
		printf("--------------------------------------------:\n");
		printf("打印误差最大的实部和虚部的值:\n");

		for(int i=max_dist_Real_ID;i<max_dist_Real_ID+1;i++)
		{
			printf("Check Real:\n");
			printf("误差最大点实部值 %f\n",i,A1[i*2]);
			printf("对应目标值 %f\n",i,FFT_Result_Real[i%(64*1024)]);
			printf("相对误差 %f\n",i,FFT_err_Real[i]);
		}
		for(int i=max_dist_Img_ID;i<max_dist_Img_ID+1;i++)
		{
			printf("Check Img:\n");
			printf("误差最大点虚部值 %f\n",i,A1[i*2+1]);
			printf("对应目标值 %f\n",i,FFT_Result_Img[i%(64*1024)]);
			printf("相对误差 %f\n",i,FFT_err_Img[i]);
		}
		/******************************将最大误差的点的值和误差都打印出来******************************/
	}

	/* Release Host memory objects */
	if (A1)
		free(A1);
	if (B1)
		free(B1);
	if (C1)
		free(C1);

	//release ocl object
	for (i = 0; i < num_platforms; ++i)
	{
		/* Release OpenCL working objects. */
		clReleaseProgram(programs_for_kernel_one[i]);
		clReleaseProgram(programs_for_kernel_two[i]);
		clReleaseContext(contexts[i]);
		for (int dev_id = 0; dev_id < num_devices[i]; ++dev_id)
		{
			/* Release OpenCL memory objects. */
			clReleaseMemObject(bufC1[i][dev_id]);
			clReleaseMemObject(bufB1[i][dev_id]);
			clReleaseMemObject(bufA1[i][dev_id]);
			clReleaseCommandQueue(cmd_queues[i][dev_id]);
		}
	}

	return ret;
}