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

	int max_it, N, *h_fu, type;
	double *h_U, *h_U_old;

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

	init_F(N, h_fu);

	#pragma omp parallel firstprivate(h_U,h_U_old) private(i,j) \
	shared(N)
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

	if (type == 1){

		double *d_U, *d_U_old, *tmp;
	    int *d_fu;

	    cudaMalloc((void **)&d_U, size_double);
	    cudaMalloc((void **)&d_U_old, size_double);
	    cudaMalloc((void **)&d_fu, size_int);

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

	    print_matrix(N,N,h_U);

		cudaFreeHost(h_U);
		cudaFreeHost(h_U_old);
		cudaFreeHost(h_fu);
		cudaFree(d_U);
		cudaFree(d_U_old);
		cudaFree(d_fu);

	} else if (type == 2){

		double *d_U, *d_U_old, *tmp;
	    int *d_fu;

	    cudaMalloc((void **)&d_U, size_double);
	    cudaMalloc((void **)&d_U_old, size_double);
	    cudaMalloc((void **)&d_fu, size_int);

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

	    print_matrix(N,N,h_U);

		cudaFreeHost(h_U);
		cudaFreeHost(h_U_old);
		cudaFreeHost(h_fu);
		cudaFree(d_U);
		cudaFree(d_U_old);
		cudaFree(d_fu);

	} else if (type == 3 && N % 2 == 0){ //N has to be an even number

		//int access0from1, access1from0, ; //access gpu 0 from gpu 1
		int flags, gpu_0 = 0, gpu_1 = 1;

		// Enable peer access
		//cudaDeviceCanAccessPeer(&access0from1, gpu_1, gpu_0);
		//cudaDeviceCanAccessPeer(&access1from0, gpu_0, gpu_1);

		cudaSetDevice(gpu_0);
		cudaDeviceEnablePeerAccess(gpu_1,flags=0);

		cudaSetDevice(gpu_1);
		cudaDeviceEnablePeerAccess(gpu_0,flags=0);

		// Do operations
		double *d0_U, *d0_U_old, *d1_U, *d1_U_old, *tmp0, *tmp1;
	    int *d0_fu, *d1_fu;

	    dim3 dimBlock(16, 16, 1); // Num threads
	    dim3 dimGrid(ceil((double)N/dimBlock.x), ceil((N/2)/dimBlock.y), 1); // Num blocks


	    // Device 0
	    cudaSetDevice(gpu_0);

	    cudaMalloc((void **)&d0_U, size_double/2.0);
	    cudaMalloc((void **)&d0_U_old, size_double/2.0);
	    cudaMalloc((void **)&d0_fu, size_int/2);

	    cudaMemcpy(d0_U,     h_U,     size_double/2.0, cudaMemcpyHostToDevice);
	    cudaMemcpy(d0_U_old, h_U_old, size_double/2.0, cudaMemcpyHostToDevice);
	    cudaMemcpy(d0_fu,    h_fu,    size_int/2,      cudaMemcpyHostToDevice);

	    // Device 1
	    cudaSetDevice(gpu_1);

	    cudaMalloc((void **)&d1_U, size_double/2.0);
	    cudaMalloc((void **)&d1_U_old, size_double/2.0);
	    cudaMalloc((void **)&d1_fu, size_int/2);

	    cudaMemcpy(d1_U,     h_U + (N*N)/2,     size_double/2.0, cudaMemcpyHostToDevice);
	    cudaMemcpy(d1_U_old, h_U_old + (N*N)/2, size_double/2.0, cudaMemcpyHostToDevice);
	    cudaMemcpy(d1_fu,    h_fu + (N*N)/2,    size_int/2,      cudaMemcpyHostToDevice);

	    while (k < max_it) {

	    	cudaSetDevice(gpu_0);
	    	jacobi_3<<<dimGrid,dimBlock>>>(N, d0_U, d0_U_old, d1_U_old, d0_fu, h, delta_sq, gpu_0);
	    	cudaDeviceSynchronize();

	    	cudaSetDevice(gpu_1);
	    	jacobi_3<<<dimGrid,dimBlock>>>(N, d1_U, d1_U_old, d0_U_old, d1_fu, h, delta_sq, gpu_1);
		    cudaDeviceSynchronize();

		    //Swap Pointers	device 0
		    tmp0 = d0_U;
		    d0_U = d0_U_old;
		    d0_U_old = tmp0;

		   	//Swap Pointers	device 1
		    tmp1 = d1_U;
		    d1_U = d1_U_old;
		    d1_U_old = tmp1;

			k++;
	    }

	    cudaMemcpy(h_U,     d0_U,     size_double/2.0, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_U_old, d0_U_old, size_double/2.0, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_fu,    d0_fu,    size_int/2,      cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_U + (N*N)/2,     d1_U,     size_double/2.0, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_U_old + (N*N)/2, d1_U_old, size_double/2.0, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_fu + (N*N)/2,    d1_fu,    size_int/2,      cudaMemcpyDeviceToHost);

	    print_matrix(N,N,h_U);

		cudaFreeHost(h_U);
		cudaFreeHost(h_U_old);
		cudaFreeHost(h_fu);
		cudaFree(d0_U);
		cudaFree(d0_U_old);
		cudaFree(d0_fu);
		cudaFree(d1_U);
		cudaFree(d1_U_old);
		cudaFree(d1_fu);





	}
		
	return(0); //end of main

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
