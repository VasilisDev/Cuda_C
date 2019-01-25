#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda.h>


/* Problem size. */
#define NX 4096
#define NY 4096

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(double *x, double *A)
{
	int i, j;

	for (i = 0; i < NX; i++) {
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++) {
			A[i*NY + j] = ((double) i*(j)) / NX;
		}
	}
}

__global__ void kernel1(int nx, int ny, double *A, double *x, double *tmp)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nx)
	{
		tmp[i] = 0;
		int j;
		for(j=0; j < ny; j++)
		{
			tmp[i] =tmp[i] + A[i*ny+j] * x[j];
		}
	}
}

__global__ void kernel2(int nx, int ny, double *A, double *y, double *tmp)
{
 unsigned	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < ny)
	{
		y[j] = 0;
		int i;
		for(i=0; i < nx; i++)
		{
			y[j] = y[j] + A[i*ny+j] * tmp[i];
		}
	}
}

void trans_norm_vector(double* A, double* x, double* y, double* tmp)
{
 	int i,j;

	for (i= 0; i < NY; i++) {
    	y[i] = 0;
	}

	for (i = 0; i < NX; i++) {
      		tmp[i] = 0;

	      	for (j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}

	      	for (j = 0; j < NY; j++) {
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
	}
}

int main(int argc, char *argv[])
{
	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	double    *gpuRef;
	struct timeval	cpu_start, cpu_end;

	// set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device properties %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

	// set up data size of matrix
  int nx = NX;
  int ny = NY;
  int nxy = nx * ny;
  size_t nBytes = nxy * sizeof(double);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	A = (double*)malloc(nBytes*sizeof(double));
	x = (double*)malloc(ny*sizeof(double));
	y = (double*)malloc(ny*sizeof(double));
	tmp = (double*)malloc(nx*sizeof(double));
	gpuRef = (double*)malloc(nx*sizeof(double));





	// initialize data at host side
	init_array(x, A);

	gettimeofday(&cpu_start, NULL);
	trans_norm_vector(A, x, y, tmp);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU trans_norm_vector Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);


	double *gpuA,*gpuX,*gpuY,*gpuTmp;

	// malloc device global memory
	cudaMalloc((void **)&gpuA, sizeof(double)*nBytes);
	cudaMalloc((void **)&gpuX, sizeof(double)*ny);
	cudaMalloc((void **)&gpuY, sizeof(double)*ny);
	cudaMalloc((void **)&gpuTmp, sizeof(double)*nx);

	// transfer data from host to device
	cudaMemcpy(gpuA, A,nBytes*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuX, x,ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuY, y,ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuTmp,tmp,nx*sizeof(double), cudaMemcpyHostToDevice);

	// invoke kernel at host side
	int dimx = 1024;
	int dimy = 2;
	dim3 block(dimx, dimy);

	//dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	//dim3 grid4 ((nx + block.x - 1) / (block.x * 4), (ny + block.y - 1) /
				//	(block.y * 4));

	dim3 grid1(nx/ block.x, 1);
	dim3 grid2(ny/ block.x, 1);

	gettimeofday(&cpu_start, NULL);
	kernel1<<< grid1, block >>>(nx, ny,gpuA,gpuX,gpuTmp);
	cudaDeviceSynchronize();
	kernel2<<< grid2, block >>>(nx, ny,gpuA,gpuY,gpuTmp);
	cudaDeviceSynchronize();
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "GPU trans_norm_vector Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// check kernel error
  cudaGetLastError();
	cudaMemcpy(gpuRef,gpuY, sizeof(double)*nx, cudaMemcpyDeviceToHost);

	// free device global memory
	cudaFree(gpuA);
	cudaFree(gpuX);
	cudaFree(gpuX);
	cudaFree(gpuTmp);

	// free host memory
	free(A);
	free(x);
	free(y);
	free(tmp);

	// reset device
	cudaDeviceReset();

  return (0);
}
