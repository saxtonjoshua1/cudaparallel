/*
 ============================================================================
 Name        : monte_integrate.cu
 Author      : Joshua Saxton Kennesaw State University
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <curand_kernel.h>

#define MAX_BLOCK_SIZE 1024
#define THREAD_ITERATIONS 64

__device__ float function(float x)
{
	return x*x;
}

void Get_args(int argc, char* argv[], float* b_input, int* threads_per_block_input, int* blocks_input)
{
   if (argc != 3) {
      fprintf(stderr, "usage: %s <b - integration limit> <blocks> <threads per block>\n",
            argv[0]);
      exit(0);
   }

   *b_input = strtod(argv[1], NULL);
   *blocks_input = strtol(argv[2], NULL, 10);
   *threads_per_block_input = strtol(argv[3], NULL, 10);
}

__global__ void integrate_kernel(float *estimate, curandState *states, float *b) {

		unsigned int thread = threadIdx.x + blockDim.x*blockIdx.x;

		int points_in = 0;
		float x, y;
		// Initialize CURAND

		curand_init(thread, 0, 0, &states[thread]);

		for(int i = 0; i < THREAD_ITERATIONS; i++) {
			x = curand_uniform(&states[thread]);
			y = curand_uniform(&states[thread]);
			// count if x & y is under the function
			if(y < function(x))
			{
				points_in++;
			}
		}
		//not sure how to compute a bounding box from 0 to b but maybe a simple square works?
		estimate[thread] =  (b * b * points_in) / (float) THREAD_ITERATIONS;

}

int main(int argc, char* argv[])
{
	int threads_per_block, blocks;
	float *b;
	float host[blocks * threads_per_block];
	float *device_result;
	curandState *devStates;

    Get_args(argc, argv, &b, &threads_per_block, &blocks);

    cudaMalloc((void **) &device_result, blocks * threads_per_block * sizeof(float));
    cudaMalloc((void **) &devStates, blocks*threads_per_block * sizeof(curandState));

    integrate_kernel<<<blocks, threads_per_block>>>(device_result, devStates,b);

    cudaMemcpy(host, device_result, blocks * threads_per_block * sizeof(float),cudaMemcpyDeviceToHost);

    // on the cpu compute a partial sum sequentially.
    float result=0.0;
    for(int i = 0; i < blocks * threads_per_block; i++)
		 result += host[i];
		 result /= (blocks * threads_per_block);
    printf("Freeing memory");
	cudaFree(device_result);
	cudaFree(devStates);
	printf("Returning result");
    return result;


}


