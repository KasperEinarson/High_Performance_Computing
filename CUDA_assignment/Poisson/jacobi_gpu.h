#ifndef __JACOBI_H
#define __JACOBI_H

__global__ void jacobi(int N, double *U, double *U_old, int *F, double h, double delta_sq);

#endif
