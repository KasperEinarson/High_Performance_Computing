#include <stdlib.h>
#include "matmult_gpu.h"

#define a(i,l) A[(i)*k + (l)]
#define b(l,j) B[(l)*blockDim.x + (j)]
#define c(i,j) C[(i)*blockDim.x + (j)]

__global__ void matmult_gpu(int k,double *A,double *B,double *C) {

    int i,j,l;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;

    for (l = 0; l<k; l++) {
       	c(i,j) = c(i,j) + a(i,l) * b(l,j);
    }

}
