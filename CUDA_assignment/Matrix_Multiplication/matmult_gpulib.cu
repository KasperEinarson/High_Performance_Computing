#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

// Declarations
extern "C" {
void matmult_gpulib(int m, int n, int k,double *h_A,double *h_B,double *h_C);
}

void matmult_gpulib(int m,int n,int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, size_C);

    cublasHandle_t handle;
    double alpha = 1.0, beta = 0.0;
    int lda = k, ldb = n, ldc = n;

    cublasCreate(&handle);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    //cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
