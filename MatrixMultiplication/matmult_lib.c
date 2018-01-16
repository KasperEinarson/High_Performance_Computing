# include <stdio.h>
# include <stdlib.h>
# include <cblas.h>

// Declarations
void matmult_lib(int m,int n,int k,double *A,double *B,double *C);

void matmult_lib(int m,int n,int k,double *A,double *B,double *C) {
  int alpha = 1, beta = 0;
  int lda = k, ldb = n, ldc = n;
  
  int i;
  for (i=0; i<m*n; i++) C[i] = 0;

  cblas_dgemm(CblasRowMajor,  CblasNoTrans, CblasNoTrans, m, n, k,
    alpha, A, lda, B, ldb, beta, C, ldc);
}
