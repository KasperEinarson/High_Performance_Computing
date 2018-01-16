#include <stdio.h>
#include <helper_cuda.h>

// Declarations
__global__ void helloworld();

int main(int argc, char *argv[])
{
  helloworld<<<8,32>>>();
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void helloworld() {
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_threads = blockDim.x * gridDim.x;

  if (global_thread_idx == 100) {
    int *a = (int*) 0x10000;
    *a = 0;
  }

  printf("Hello world! I'm thread %d out of %d in block %d. My global thread id is %d out of %d\n",
    threadIdx.x, blockDim.x, blockIdx.x, global_thread_idx, global_threads);
}
