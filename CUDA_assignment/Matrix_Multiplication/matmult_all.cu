#include <stdlib.h>
#include <stdio.h>

#define a(i,l) A[(i)*k + (l)]
#define b(l,j) B[(l)*n + (j)]
#define c(i,j) C[(i)*n + (j)]
#define BLOCK_SIZE 16
#define num_el 4

// Declarations
extern "C" {
void matmult_gpu1(int m, int n, int k,double *h_A,double *h_B,double *h_C);
void matmult_gpu2(int m, int n, int k,double *h_A,double *h_B,double *h_C);
void matmult_gpu3(int m, int n, int k,double *h_A,double *h_B,double *h_C);
void matmult_gpu4(int m, int n, int k,double *h_A,double *h_B,double *h_C);
void matmult_gpu5(int m, int n, int k,double *h_A,double *h_B,double *h_C);
}

__global__ void matmult1(int m, int n, int k,double *A,double *B,double *C);
__global__ void matmult2(int m, int n, int k,double *A,double *B,double *C);
__global__ void matmult3(int m, int n, int k,double *A,double *B,double *C);
__global__ void matmult4(int m, int n, int k,double *A,double *B,double *C);
__global__ void matmult5(int m, int n, int k,double *A,double *B,double *C);

void matmult_gpu1(int m, int n, int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    dim3 dimBlock(1, 1, 1); // Num threads
    dim3 dimGrid(1, 1, 1); // Num blocks

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, size_C);

    matmult1<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

__global__ void matmult1(int m, int n, int k,double *A,double *B,double *C) {

    int i,j,l;

    for (i = 0; i<m; i++) {
        for (l = 0; l<k; l++) {
            for (j = 0; j<n; j++) {
             	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            }
        }
    }

}

void matmult_gpu2(int m, int n, int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    dim3 dimBlock(16, 16, 1); // Num threads
    dim3 dimGrid(ceil((double)n/dimBlock.x), ceil((double)m/dimBlock.y), 1); // Num blocks

    //printf("x: %d, y: %d, z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, size_C);

    matmult2<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

__global__ void matmult2(int m, int n, int k,double *A,double *B,double *C) {

    int i,j,l;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;

    double C_reg = 0.0;

    if (i < m && j < n) {
      for (l = 0; l<k; l++) {
         	C_reg = C_reg + a(i,l) * b(l,j);
      }
      c(i,j) = C_reg;
    }
}

void matmult_gpu3(int m, int n, int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    dim3 dimBlock(16, 16, 1); // Num threads
    dim3 dimGrid((ceil((double)n/dimBlock.x)), ceil(((double)m/dimBlock.y) / 2), 1); // Num blocks

    //printf("x: %d, y: %d, z: %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, size_C);

    matmult3<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

__global__ void matmult3(int m, int n, int k,double *A,double *B,double *C) {

    int i,j,l;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    double C_reg[2] = {0.0};


    if (i < m-1 && j < n) {
      for (l = 0; l<k; l++) {
         	C_reg[0] = C_reg[0] + a(i,l) * b(l,j);
            C_reg[1] = C_reg[1] + a(i+1,l) * b(l,j);
      }
      //Copy back to global memory
      c(i,j) = C_reg[0];
      c(i+1,j) = C_reg[1];

    } else if (i == m-1 && j < n) {
        for (l = 0; l<k; l++) {
           	C_reg[0] = C_reg[0] + a(i,l) * b(l,j);
        }
        //Copy back to global memory
        c(i,j) = C_reg[0];
    }

    /*
    j = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n-1) {
      for (l = 0; l<k; l++) {
         	c(i,j) = c(i,j) + a(i,l) * b(l,j);
            c(i,j+1) = c(i,j+1) + a(i,l) * b(l,j+1);
      }
    } else if (i < m && j == n-1) {
        for (l = 0; l<k; l++) {
           	c(i,j) = c(i,j) + a(i,l) * b(l,j);
        }
    }
    */

}

void matmult_gpu4(int m, int n, int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1); // Num threads
    dim3 dimGrid((ceil((double)n/dimBlock.x)), ceil(((double)m/dimBlock.y) / num_el), 1); // Num blocks

    cudaMemset(d_C, 0, size_C);

    matmult4<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

__global__ void matmult4(int m, int n, int k,double *A,double *B,double *C) {

    int i,j,l,s;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = (blockIdx.y * blockDim.y + threadIdx.y) * num_el;

    double C_reg[num_el] = {0.0};

    if (i < m-num_el && j < n) {
      for (l = 0; l<k; l++) {
          C_reg[0] = C_reg[0] + a(i,l) * b(l,j);
          C_reg[1] = C_reg[1] + a(i+1,l) * b(l,j);
          C_reg[2] = C_reg[2] + a(i+2,l) * b(l,j);
          C_reg[3] = C_reg[3] + a(i+3,l) * b(l,j);
      }
      c(i,j) = C_reg[0];
      c(i+1,j) = C_reg[1];
      c(i+2,j) = C_reg[2];
      c(i+3,j) = C_reg[3];

    } else if (i >= m-num_el && j < n) {
        for (s = i; s<m; s++) {
            for (l = 0; l<k; l++) {
                C_reg[s-i] = C_reg[s-i] + a(s,l) * b(l,j);
            }
            c(s,j) = C_reg[s-i];
        }
    }

}


/*
void matmult_gpu5(int m, int n, int k,double *h_A,double *h_B,double *h_C) {

    double  *d_A, *d_B, *d_C;
    int size_A = m * k * sizeof(double);
    int size_B = k * n * sizeof(double);
    int size_C = m * n * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1); // Num threads
    dim3 dimGrid((ceil((double)n/dimBlock.x)), ceil(((double)m/dimBlock.y) / num_el), 1); // Num blocks

    cudaMemset(d_C, 0, size_C);

    matmult4<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

__global__ void matmult5(int m, int n, int k,double *A,double *B,double *C) {

    int i,j,l,s;

    double Cvalue = 0;
    double * Csub;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    k_blocked = (k/BLOCK_SIZE)

    //Get Csub matrix
    Csub = &A[k*BLOCK_SIZE * j+ BLOCK_SIZE * i]

    if (i < m-num_el && j < n) {
      for (l = 0; l<k_blocked; l++) {


          //Shared memory to store sub-matrices of A and B
          __shared__ double as[BLOCK_SIZE][BLOCK_SIZE];
          __shared__ double bs[BLOCK_SIZE][BLOCK_SIZE];

          as[threadIdx.y][threadIdx.x] = A[i*k + l*BLOCK_SIZE + threadIdx.x];
          bs[threadIdx.y][threadIdx.x] = B[(l*BLOCK_SIZE+threadIdx.y)*];

          __syncthreads();
          //Multiply sub matrices
          for(int e = 0; e < BLOCK_SIZE; ++e){
              Cvalue += as[i][e] * bs[e][l]
          }
          __syncthreads();

      }
    } else if (i >= m-num_el && j < n) {
        for (l = 0; l<k; l++) {
            for (s = i; s<m; s++) {
                c(s,j) = c(s,j) + a(s,l) * b(l,j);
            }
        }
    }

    //Write back to global memory somehow


}
*/
