#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda.h>



/* Problem size */
#define NI 4096
#define NJ 4096

void Convolution(double* A, double* B)
{
	int i, j;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) {
		for (j = 1; j < NJ - 1; ++j) {
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)]
				    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}

__global__ void convolutionOnGPU(int nx, int ny, double *MatA, double *MatB)
{
unsigned	int j = blockIdx.x * blockDim.x + threadIdx.x;
unsigned	int i = blockIdx.y * blockDim.y + threadIdx.y;

	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		MatB [i * NJ + j] =  c11 * MatA[(i - 1) * NJ + (j - 1)]  + c21 * MatA[(i - 1) * NJ + (j + 0)] + c31 * MatA[(i - 1) * NJ + (j + 1)]
			+ c12 * MatA[(i + 0) * NJ + (j - 1)]  + c22 * MatA[(i + 0) * NJ + (j + 0)] +  c32 * MatA[(i + 0) * NJ + (j + 1)]
			+ c13 * MatA[(i + 1) * NJ + (j - 1)]  + c23 * MatA[(i + 1) * NJ + (j + 0)] +  c33 * MatA[(i + 1) * NJ + (j + 1)];
	}
}


void checkResult(double *hostRef, double *gpuRef, const int N)
{
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = 0;
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

void init(double* A)
{
	int i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{

 printf("2D convolution starting...\n");

 // set up device
 int dev = 0;
 cudaDeviceProp deviceProp;
 cudaGetDeviceProperties(&deviceProp, dev);
 printf("Device properties %d: %s\n", dev, deviceProp.name);
 cudaSetDevice(dev);

 struct timeval	cpu_start, cpu_end;

	// set up data size of matrix
 int nx = NI;
 int ny = NJ;
 int nxy = nx * ny;
 size_t nBytes = nxy * sizeof(double);
 printf("Matrix size: nx %d ny %d\n", nx, ny);

 // malloc host memory
 double		*A,*B,*gpuRef;

	A = (double*)malloc(nBytes);
	B = (double*)malloc(nBytes);
	gpuRef = (double*)malloc(nBytes);



	//initialize the arrays
	init(A);

	memset(gpuRef, 0, nBytes);


	gettimeofday(&cpu_start, NULL);
	Convolution(A, B);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "Convolution time on host: %0.6lf sec\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);



	double *d_MatA,*d_MatB;
	// malloc device global memory
	cudaMalloc((void **)&d_MatA,nBytes);
	cudaMalloc((void **)&d_MatB,nBytes);


	// transfer data from host to device
	cudaMemcpy(d_MatA, A ,nBytes,cudaMemcpyHostToDevice);


	// invoke kernel at host side
	int dimx = 1024;
	int dimy = 2;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
				 block.y);

	gettimeofday(&cpu_start, NULL);
	convolutionOnGPU <<< grid,block >>> (nx, ny, d_MatA,d_MatB);
	gettimeofday(&cpu_end, NULL);

  cudaDeviceSynchronize();

	printf("Convolution time on gpu: %0.6lf sec\n",((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);


	// check kernel error
  cudaGetLastError();

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_MatB, nBytes, cudaMemcpyDeviceToHost);

	// free device global memory
	cudaFree(d_MatA);
  cudaFree(d_MatB);

	// check device results
	checkResult(B, gpuRef, nxy);

	// free host memory
	free(A);
	free(B);
	free(gpuRef);

	// reset device
	cudaDeviceReset();


	return (0);
}
