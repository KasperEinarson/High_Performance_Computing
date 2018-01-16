#include <stdio.h>
#include <stdlib.h>
#include "matmult_gpu.h"

// Declarations
void print_matrix(int m, int n, double * matrix);

int main(int argc, char *argv[]) {

    int   m, n, k, i;
    double   *h_A, *d_A, *h_B, *d_B, *h_C, *d_C;

    m = atoi(argv[1]);
  	k = atoi(argv[2]);
  	n = atoi(argv[3]);

    cudaMallocHost((void **)&h_A, m * k * sizeof(double));
    cudaMallocHost((void **)&h_B, k * n * sizeof(double));
    cudaMallocHost((void **)&h_C, m * n * sizeof(double));
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    for (i=0; i<m*k; i++) h_A[i] = i+1.0;
    for (i=0; i<k*n; i++) h_B[i] = i+1.0;
    for (i=0; i<m*n; i++) h_C[i] = 0.0;

    print_matrix(m, k, h_A);
    printf("\n");
    print_matrix(k, n, h_B);
    printf("\n");

    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1); // Num threads
    dim3 dimGrid(ceil((double)m/dimBlock.x), ceil((double)n/dimBlock.y), 1); // Num blocks

    matmult_gpu<<<dimGrid,dimBlock>>>(k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    print_matrix(m, n, h_C);

    cudaFreeHost(h_A);
    cudaFree(d_A);
    cudaFreeHost(h_B);
    cudaFree(d_B);
    cudaFreeHost(h_C);
    cudaFree(d_C);


    return(0);
}

void print_matrix(int m, int n, double * matrix) {
  int x = 0;
  int y = 0;

  for(x = 0 ; x < m ; x++) {
    printf(" (");
    for(y = 0 ; y < n ; y++){
      printf("%f     ", matrix[n * x + y]);
    }
    printf(")\n");
  }
}
