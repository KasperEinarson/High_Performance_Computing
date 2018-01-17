#ifndef __JACOBI_H
#define __JACOBI_H

__global__ void jacobi_1(int N, double *U, double *U_old, int *F, double h, double delta_sq);
__global__ void jacobi_2(int N, double *U, double *U_old, int *F, double h, double delta_sq);

#endif
