/*
 ============================================================================
 Name        : assignment5.cu
 Author      : Joshua Saxton Kennesaw State University
 Version     :
 Copyright   : Your copyright notice
 Description : Trapezoid rule for cuda
 Compile:  nvcc assignment5.cu -o ass5
 Run:      ./ass5 <n> <a> <b> <blocks> <threads_per_block>
 	 	 max threads per block is 1024
           n is the number of trapezoids
           a is the left endpoint
           b is the right endpoint
 ============================================================================
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define MAX_BLOCK_SIZE 1024

__device__ float function(float x)
{
	return x*x + 1;
}

void Get_args(int argc, char* argv[], int* n_input, float* a_input, float* b_input,int* threads_per_block_input, int* blocks_input)
{
   if (argc != 6) {
      fprintf(stderr, "usage: %s <n> <a> <b> <blocks> <threads per block>\n",
            argv[0]);
      exit(0);
   }
   *n_input = strtol(argv[1], NULL, 10);
   *a_input = strtod(argv[2], NULL);
   *b_input = strtod(argv[3], NULL);
   *blocks_input = strtol(argv[4], NULL, 10);
   *threads_per_block_input = strtol(argv[5], NULL, 10);
}

__global__ void trapezoid(float a, float b, float h, int n, float z[]) {

	__shared__ float tmp[MAX_BLOCK_SIZE];

   int thread = blockDim.x * blockIdx.x + threadIdx.x;
   int local_thread = threadIdx.x;
   float area = thread*h;

   if (thread < n)
	   tmp[local_thread] = 0.5*h*(function(area) + function(area+h));
   __syncthreads();

   //reduction tree
   //finish one block, then go to the next one.
   for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
      if (local_thread < stride)
         tmp[local_thread] += tmp[local_thread + stride];
      __syncthreads();
   }

   /* Store the result in global memory */
   if (threadIdx.x == 0)
	   z[blockIdx.x] = tmp[0];
}

int main(int argc, char* argv[])
{
	int n, threads_per_block, blocks;
    float a, b, *z_d;

    Get_args(argc, argv, &n, &a, &b, &threads_per_block, &blocks);

    //make enough memory for all the blocks to hold one trapezoid
    cudaMalloc(&z_d, blocks*sizeof(float));

    //since the tmp variable is in shared memory? There is no need to malloc for the initial array

    int i;
    float result = 0.0, h;
    float z_h[blocks];

    /* Invoke kernel */
    h = (b-a)/n;
    trapezoid<<<blocks, threads_per_block>>>(a, b, h, n, z_d);
    cudaThreadSynchronize();

    cudaMemcpy(&z_h, z_d, blocks*sizeof(float), cudaMemcpyDeviceToHost);

    for (i = 0; i < blocks; i++)
       result += z_h[i];
    return result;
}



