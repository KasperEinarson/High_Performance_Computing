#include <stdio.h>
#include <stdlib.h>
extern "C" {
    #include <cblas.h>
}

// Declarations
extern "C" {
void matmult_lib(int m,int n,int k,double *A,double *B,double *C);
}

void matmult_lib(int m,int n,int k,double *A,double *B,double *C) {
  int alpha = 1, beta = 0;
  int lda = k, ldb = n, ldc = n;

  cblas_dgemm(CblasRowMajor,  CblasNoTrans, CblasNoTrans, m, n, k,
    alpha, A, lda, B, ldb, beta, C, ldc);
}
