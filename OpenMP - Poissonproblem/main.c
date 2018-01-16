#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "jacobi.h"
#include "gauss_seidel.h"
#include "jacobi_omp.h"
#include "jacobi_analytical.h"
#include "gauss_seidel_analytical.h"

#define f(i,j) F[(i)*N + (j)]
#define u(i,j) U[(i)*N + (j)]
#define u_old(i,j) U_old[(i)*N + (j)]
#define u_true(i,j) U_true[(i)*N + (j)]

// Declarations
void init_F(int N, int *F);
void init_F_analytical(int N, double *F);
void print_matrix(int m, int n, double *matrix);
void print_F(int m, int n, int *mat);
void init_U_true(int N, double *U);

int main(int argc, char *argv[]) {

	int max_it, N, typ, *fu;
	double tol, *U, *U_old, *f_analytical, *U_true;

	N = atoi(argv[1]);
	max_it = atoi(argv[2]);
	tol = atof(argv[3]);
	typ = atoi(argv[4]);

	U = (double *)malloc(N * N * sizeof(double));
	U_old = (double *)malloc(N * N * sizeof(double));
	fu = (int *)malloc(N * N * sizeof(int));
	f_analytical = (double *)malloc(N * N * sizeof(double));
	U_true = (double *)malloc(N * N * sizeof(double));
  if (U == NULL || U_old == NULL || fu == NULL || f_analytical == NULL || U_true == NULL) {
     fprintf(stderr, "memory allocation failed!\n");
     return(1);
  }

	init_F(N, fu);

	init_F_analytical(N, f_analytical);
	init_U_true(N, U_true);

	if (typ == 1) {
		jacobi(N, max_it, tol, U, U_old, fu);
	} else if (typ == 2) {
		gauss_seidel(N, max_it, tol, U, fu);
	} else if (typ == 3) {
		jacobi_omp(N, max_it, tol, U, U_old, fu);
	} else if (typ == 4) {
		jacobi_analytical(N, max_it, tol, U, U_old, f_analytical, U_true);
	} else if (typ == 5) {
		gauss_seidel_analytical(N, max_it, tol, U, f_analytical, U_true);
	} else {
		fprintf(stderr, "invalid typ!\n");
		return(1);
	}

	//print_matrix(N,N,U);

	free(U);
	free(U_old);
	free(fu);
	free(f_analytical);
	free(U_true);

	return(0);
}

void init_F(int N, int *F) {

	double delta = 2/((double)N - 1.0);
	double fx, fy;

	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {

	    fx = (double)j*delta-1.0;
	    fy = -1.0*((double)i*delta-1.0);

			if (0<=fx && fx<=(1.0/3.0) && (-2.0/3.0)<=fy && fy<=(-1.0/3.0)) {
					f(i,j) = 200;
	    } else {
	        f(i,j) = 0;
	    }
		}
  }
}

void init_F_analytical(int N, double *F) {

	double delta = 2/((double)N - 1.0);
	double pi_sq = M_PI * M_PI;
	double x, y;

	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			x = (double)j*delta-1.0;
			y = -1.0*((double)i*delta-1.0);
			f(i,j) = 2 * pi_sq * sin(M_PI * x) * sin(M_PI * y);
		}
  }
}

void init_U_true(int N, double *U_true) {

	double delta = 2/((double)N - 1.0);
	double x, y;

	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			x = (double)j*delta-1.0;
			y = -1.0*((double)i*delta-1.0);
			u_true(i,j) = sin(M_PI * x) * sin(M_PI * y);
		}
  }
}

void print_matrix(int m, int n, double *mat) {
  int x = 0;
  int y = 0;

  for(x = 0 ; x < m ; x++) {
    printf(" ");
    for(y = 0 ; y < n ; y++){
      printf("%f     ", mat[x*m+y]);
    }
    printf(";\n");
  }
}

void print_F(int m, int n, int *mat) {
  int x = 0;
  int y = 0;

  for(x = 0 ; x < m ; x++) {
    printf(" ");
    for(y = 0 ; y < n ; y++){
      printf("%d     ", mat[x*m+y]);
    }
    printf("\n");
  }
}
