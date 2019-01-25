#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>



/* Problem size */
#define M 1024
#define N 1024
#define BDIMX 16
#define BDIMY 16
#define FLOAT_N 3214212.01

void init_arrays(double* data)
{
	int i, j;

	for (i = 1; i < (M+1); i++) {
		for (j = 1; j < (N+1); j++) {
			data[i*(N+1) + j] = ((double) i*j) / M;
		}
	}
}

void covariance(double* data, double* symmat, double* mean)
{
	int	i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j < (M+1); j++) {
		mean[j] = 0.0;
		for (i = 1; i < (N+1); i++) {
        		mean[j] += data[i*(M+1) + j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 1; i < (N+1); i++) {
		for (j = 1; j < (M+1); j++) {
			data[i*(M+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 < (M+1); j1++) {
		for (j2 = j1; j2 < (M+1); j2++) {
	       		symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i < N+1; i++) {
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
        		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
      		}
	}
}

__global__ void kernel1(int m, int n, double *mean, double *data)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < M)
	{
		mean[j] = 0.0;

		int i;
		for(i = 0; i < N; i++)
		{
			mean[j] += data[i * M + j];
		}
		mean[j] /= FLOAT_N;
	}
}

__global__ void kernel2(int m, int n,double *mean, double *data)
{
unsigned	int j = blockIdx.x * blockDim.x + threadIdx.x;
unsigned	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j <M))
	{
		data[i * M + j] -= mean[j];
	}
}

__global__ void kernel3(int m, int n,double *symmat,double *data)
{
	unsigned int j1 = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j2;

	if (j1 < M)
	{
		for (j2 = j1; j2 < M; j2++)
		{
			symmat[j1*M + j2] = 0.0;
			for(i = 0; i < N; i++)
			{
				symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
			}
			symmat[j2 * M + j1] = symmat[j1 * M + j2];
		}
	}
}

int main(int argc, char *argv[])
{
	double		*data;
	double		*symmat;
	double		*mean;
	unsigned int n=N,m=N;

	printf("covariance starting...\n");

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device properties %d: %s\n", dev, deviceProp.name);
	cudaSetDevice(dev);

	struct timeval	cpu_start, cpu_end;

	 data = (double*)malloc((M+1)*(N+1)*sizeof(double));
	 symmat = (double*)malloc((M+1)*(M+1)*sizeof(double));
   mean = (double*)malloc((M+1)*sizeof(double));

	 init_arrays(data);

	 gettimeofday(&cpu_start, NULL);
	 covariance(data, symmat, mean);
	 gettimeofday(&cpu_end, NULL);
	 fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	 double *dataOnGPU;
	 double *meanOnGPU;
	 double *symmatOnGPU;
  	// malloc device global memory
   cudaMalloc((void **)&dataOnGPU,(M+1)*(N+1)*sizeof(double));
	 cudaMalloc((void **)&symmatOnGPU,(M+1)*(M+1)*sizeof(double));
	 cudaMalloc((void **)&meanOnGPU,(M+1)*sizeof(double));

	 // transfer data from host to device
	 cudaMemcpy(dataOnGPU,data,(M+1)*(N+1)*sizeof(double),cudaMemcpyHostToDevice);
	 cudaMemcpy(symmatOnGPU,symmat,(M+1)*(M+1)*sizeof(double),cudaMemcpyHostToDevice);
	 cudaMemcpy(meanOnGPU, mean,(M+1)*sizeof(double),cudaMemcpyHostToDevice);

	  // invoke kernel at host side
		dim3 block (BDIMX, BDIMY);
		dim3 grid  (1, 1);
		printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
					 block.y);

		gettimeofday(&cpu_start, NULL);
	  kernel1<<<grid, block>>>(m,n,meanOnGPU,dataOnGPU);
		cudaDeviceSynchronize();
		kernel2<<<grid, block>>>(m,n,meanOnGPU,dataOnGPU);
		cudaDeviceSynchronize();
		kernel3<<<grid, block>>>(m,n,symmatOnGPU,dataOnGPU);
		cudaDeviceSynchronize();
		gettimeofday(&cpu_end, NULL);
	  fprintf(stdout, "GPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	  // check kernel error
  	cudaGetLastError();

	  // free device global memory
	  cudaFree(dataOnGPU);
    cudaFree(symmatOnGPU);
    cudaFree(meanOnGPU);

	   // free host memory
	  free(data);
	  free(symmat);
	  free(mean);
	  // reset device
 	  cudaDeviceReset();

 	  return (0);
}
