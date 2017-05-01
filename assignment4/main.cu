#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

using namespace std;

void fatal_error (const char* message);
// precondition: message is not NULL
// postcondition: message has been written to standard error & program
//                terminated

char* create_error_string (const char* message, const char* data_string);


__global__ void histogram(unsigned int *input, unsigned int *bins,
		unsigned int num_elements, unsigned int num_bins);
  
int main (int argc, char *argv[] )
{
  if (argc != 2)
    fatal_error("assn2 <num_intervals> <file\n");
  
  int num_intervals= argv[1];
  char file_name = argv[2];

  	streampos size;
    char * memblock;

    ifstream file (file_name, ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
      size = file.tellg();
      memblock = new char [size];
      file.seekg (0, ios::beg);
      file.read (memblock, size);
      file.close();

      cout << "the entire file content is in memory";

    }
    else cout << "Unable to open file";

  return 0;
}

void fatal_error (const char* message)
{
  fprintf (stderr, "%s", message);
  exit (0);
}

char* create_error_string (const char* message, const char* data_string)
{
  char* result = (char*)malloc (60 * sizeof(char));
  if (result == NULL)
    fatal_error ("malloc error");
  snprintf (result, 60, message, data_string);
  return result;
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
		unsigned int num_elements, unsigned int num_bins)
{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		// total number of threads
		int stride = blockDim.x * gridDim.x;

		if (threadIdx.x < num_bins) {
			histo_private[threadIdx.x] = 0;
		}

		__syncthreads();

		// compute block's histogram

		while (i < num_elements) {
			int temp = input[i];
			atomicAdd(&(histo_private[temp]), 1);
			i += stride;
		}
		// wait for all other threads in the block to finish
		__syncthreads();

		// store to global histogram

		if (threadIdx.x < num_bins) {
			atomicAdd(&(bins[threadIdx.x]), histo_private[threadIdx.x]);
		}

}


