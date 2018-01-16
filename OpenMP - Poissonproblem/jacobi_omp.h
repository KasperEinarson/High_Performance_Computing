#ifndef __JACOBI_OMP_H
#define __JACOBI_OMP_H

void jacobi_omp(int N, int max_it, double tol, double * U, double *U_old, int * F);

#endif
