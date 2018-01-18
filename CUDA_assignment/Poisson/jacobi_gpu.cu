#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "jacobi_gpu.h"

#define u(i,j) U[(i)*N + (j)]
#define u_old(i,j) U_old[(i)*N + (j)]
#define u_old_e(i,j) U_old_e[(i)*N + (j)]
#define f(i,j) F[(i)*N + (j)]


__global__ void jacobi_1(int N, double *U, double *U_old, int *F, double h, double delta_sq) { 

    int i,j;

    // Update U
    for (i=1; i<N-1; i++) {
      for (j=1; j<N-1; j++) {
        u(i,j) = h * (u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + delta_sq * (double)f(i,j)); 
      }
    }

}

__global__ void jacobi_2(int N, double *U, double *U_old, int *F, double h, double delta_sq) { 

  int i, j;


    j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i < N-1 && j < N-1){

    u(i,j) = h * (u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + delta_sq * (double)f(i,j));

    }
    //Swap Pointers 

}

__global__ void jacobi_3_0(int N, double *U, double *U_old, double *U_old_e, int *F, double h, double delta_sq) {

  int i,j;

    j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    i = blockIdx.y * blockDim.y + threadIdx.y + 1;

      if(i < (N/2)-1 && j < N-1){

          u(i,j) = h * (u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + delta_sq * (double)f(i,j));
          
      } else if(i == (N/2) && j < N-1){

            u(i-1,j) = h * (u_old(i-2,j) + u_old_e(0,j) + u_old(i-1,j-1) + u_old(i-1,j+1) + delta_sq * (double)f(i-1,j));

          }
}

__global__ void jacobi_3_1(int N, double *U, double *U_old, double *U_old_e, int *F, double h, double delta_sq) {

  int i,j;

    j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    i = blockIdx.y * blockDim.y + threadIdx.y + 1;

      if(i < (N/2)-1 && j < N-1){

        if(i == 1){
        
          u(i-1,j) = h * (u_old_e((N/2)-1,j) + u_old(i,j) + u_old(i-1,j-1) + u_old(i-1,j+1) + delta_sq * (double)f(i-1,j));

        }    

        u(i,j) = h * (u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + delta_sq * (double)f(i,j));
    }
}


  /* int i, j, k = 0;
  double h = 1.0 / 4.0;
  double delta = 2/((double)N - 1.0), delta_sq = delta * delta;
  double *tmp;


  double * U_gpu, * U_old_gpu, * F_gpu;




  #pragma omp parallel firstprivate(k,U,U_old) private(i,j,tmp) \
    shared(N, max_it, F, h, delta, delta_sq)
  {

  // Initialize U and U_old
  #pragma omp for
  for (i=0; i<N; i++) {
    u_old(i,0) = 20.0;
    u_old(0,i) = 20.0;
    u_old(i,N-1) = 20.0;

    u(i,0) = 20.0;
    u(0,i) = 20.0;
    u(i,N-1) = 20.0;
     
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
} */





