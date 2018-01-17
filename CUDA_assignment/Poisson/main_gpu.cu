#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "jacobi_gpu.h"


#define d_u(i,j) d_U[(i)*N + (j)]
#define d_u_old(i,j) d_U_old[(i)*N + (j)]

#define h_u(i,j) h_U[(i)*N + (j)]
#define h_u_old(i,j) h_U_old[(i)*N + (j)]

#define f(i,j) F[(i)*N + (j)]

// Declarations
void init_F(int N, int *F);
void print_matrix(int m, int n, double *matrix);
void print_F(int m, int n, int *mat);



int main(int argc, char *argv[]) {

	int max_it, N, *h_fu, *d_fu, type;
	double *h_U, *h_U_old, *d_U, *d_U_old, *tmp;

    int i, j, k = 0;
  	double h = 1.0 / 4.0;

	N = atoi(argv[1]);
	max_it = atoi(argv[2]);
	type = atoi(argv[3]);


	double delta = 2/((double)N - 1.0), delta_sq = delta * delta;

	//printf("%d", N);
	//printf("%d", max_it);

	//U = (double *)malloc(N * N * sizeof(double));
	//U_old = (double *)malloc(N * N * sizeof(double));
	//fu = (int *)malloc(N * N * sizeof(int));

	double size_double = N * N * sizeof(double);
	int size_int = N * N * sizeof(int);

    cudaMallocHost((void **)&h_U, size_double);
    cudaMallocHost((void **)&h_U_old, size_double);
    cudaMallocHost((void **)&h_fu, size_int);

    cudaMallocHost((void **)&d_U, size_double);
    cudaMallocHost((void **)&d_U_old, size_double);
    cudaMallocHost((void **)&d_fu, size_int);


	init_F(N, d_fu);

	  #pragma omp parallel firstprivate(k,d_U,d_U_old) private(i,j,tmp) \
	    shared(N, max_it, d_fu, h, delta, delta_sq)
	  {

	  // Initialize U and U_old
	  #pragma omp for
		  for (i=0; i<N; i++) {
		    h_u_old(i,0) = 20.0;
		    h_u_old(0,i) = 20.0;
		    h_u_old(i,N-1) = 20.0;

		    h_u(i,0) = 20.0;
		    h_u(0,i) = 20.0;
		    h_u(i,N-1) = 20.0;
		  }
	  #pragma omp for
		  for (i=1; i<N; i++) {
		    for (j=1; j<N-1; j++) {
		      h_u_old(i,j) = 0.0;
		      h_u(i,j) = 0.0;
		    }
		  }
	  } // end omp parallel

	//print_matrix(N,N,h_U_old);

	if(type == 1){

	dim3 dimBlock(1,1,1);
	dim3 dimGrid(1,1,1);

	cudaMemcpy(d_U,     h_U,     size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_old, h_U_old, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fu,    h_fu,    size_int,    cudaMemcpyHostToDevice);

	while (k < max_it) {

    	jacobi_1<<<dimGrid,dimBlock>>>(N, d_U, d_U_old, d_fu, h, delta_sq);
    	cudaDeviceSynchronize();

    	//Swap Pointers	
    	tmp = d_U;
    	d_U = d_U_old;
    	d_U_old = tmp;

		k++;

	}

    cudaMemcpy(h_U,     d_U,     size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U_old, d_U_old, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fu,    d_fu,    size_int,    cudaMemcpyDeviceToHost);


} else if (type == 2){

		dim3 dimBlock(16, 16, 1); // Num threads
    	dim3 dimGrid(ceil((double)N/dimBlock.x), ceil((double)N/dimBlock.y), 1); // Num blocks

		cudaMemcpy(d_U,     h_U,     size_double, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_U_old, h_U_old, size_double, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_fu,    h_fu,    size_int,    cudaMemcpyHostToDevice);

    	while (k < max_it) {

	    	jacobi_2<<<dimGrid,dimBlock>>>(N, d_U, d_U_old, d_fu, h, delta_sq);
	    	cudaDeviceSynchronize();

	    	//Swap Pointers	
	    	tmp = d_U;
	    	d_U = d_U_old;
	    	d_U_old = tmp;

			k++;

    	}

    	cudaMemcpy(h_U,     d_U,     size_double, cudaMemcpyDeviceToHost);
    	cudaMemcpy(h_U_old, d_U_old, size_double, cudaMemcpyDeviceToHost);
    	cudaMemcpy(h_fu,    d_fu,    size_int,    cudaMemcpyDeviceToHost);

	}

	print_matrix(N,N,h_U);

	cudaFreeHost(h_U);
	cudaFreeHost(h_U_old);
	cudaFreeHost(h_fu);
	cudaFree(d_U);
	cudaFree(d_U_old);
	cudaFree(d_fu);
	

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
