#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "jacobi_gpu.h"

#define chunk 50

#define u(i,j) U[(i)*N + (j)]
#define u_old(i,j) U_old[(i)*N + (j)]
#define f(i,j) F[(i)*N + (j)]

void jacobi_gpu(int N, int max_it, double *U, double *U_old, int *F) {

  int i, j, k = 0;
  double h = 1.0 / 4.0;
  double delta = 2/((double)N - 1.0), delta_sq = delta * delta;
  double *tmp;

  #pragma omp parallel firstprivate(k,U,U_old) private(i,j,tmp) \
    shared(N, max_it, F, h, delta, delta_sq)
  {

  // Initialize U and U_old
  #pragma omp for
  for (i=0; i<N; i++) {
    u_old(i,0) = 20.0;
    u_old(0,i) = 20.0;
    u_old(i,N-1) = 20.0;
  }
  #pragma omp for
  for (i=1; i<N; i++) {
    for (j=1; j<N-1; j++) {
      u_old(i,j) = 0.0;
    }
  }
  } // end omp parallel

  while (k < max_it) {

 
    // Update U
    for (i=1; i<N-1; i++) {
      for (j=1; j<N-1; j++) {
        u(i,j) = h * (u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + delta_sq * (double)f(i,j));	
      }
    }
   
    //Swap Pointers	
    tmp = U;
    U = U_old;
    U_old = tmp;


    k++;
  }

  

}
