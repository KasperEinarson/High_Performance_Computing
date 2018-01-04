#include <math.h>
#include <stdlib.h>

#define a(i,l) A[(i)*k + (l)]
#define b(l,j) B[(l)*n + (j)]
#define c(i,j) C[(i)*n + (j)]

// Declarations
void matmult_mnk(int m,int n,int k,double *A,double *B,double *C);
void matmult_mkn(int m,int n,int k,double *A,double *B,double *C);
void matmult_nmk(int m,int n,int k,double *A,double *B,double *C);
void matmult_nkm(int m,int n,int k,double *A,double *B,double *C);
void matmult_kmn(int m,int n,int k,double *A,double *B,double *C);
void matmult_knm(int m,int n,int k,double *A,double *B,double *C);

void matmult_mnk(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (i = 0; i<m; i++) {
        for (j = 0; j<n; j++) {
            for (l = 0; l<k; l++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_mkn(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (i = 0; i<m; i++) {
        for (l = 0; l<k; l++) {
            for (j = 0; j<n; j++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_nmk(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (j = 0; j<n; j++) {
        for (i = 0; i<m; i++) {
            for (l = 0; l<k; l++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_nkm(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (j = 0; j<n; j++) {
        for (l = 0; l<k; l++) {
            for (i = 0; i<m; i++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_kmn(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (l = 0; l<k; l++) {
        for (i = 0; i<m; i++) {
            for (j = 0; j<n; j++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_knm(int m,int n,int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i=0; i<m*n; i++) C[i] = 0;

    for (l = 0; l<k; l++) {
        for (j = 0; j<n; j++) {
            for (i = 0; i<m; i++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}
