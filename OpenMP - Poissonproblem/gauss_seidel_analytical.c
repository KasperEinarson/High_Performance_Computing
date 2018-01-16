#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "gauss_seidel_analytical.h"

#define u(i,j) U[(i)*N + (j)]
#define f(i,j) F[(i)*N + (j)]
#define u_true(i,j) U_true[(i)*N + (j)]

void gauss_seidel_analytical(int N, int max_it, double tol, double *U, double *F, double *U_true) {

  int i, j, k = 0;
  double h = 1.0 / 4.0;
  double delta = 2/((double)N - 1.0), delta_sq = delta * delta;
  double d = tol + 1.0;
  double prev = 0.0;
  double diff;

  // Initialize U
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      u(i,j) = 0.0;
    }
  }

  while (d > tol && k < max_it) {
    d = 0.0;

    // Update U
    for (i=1; i<N-1; i++) {
      for (j=1; j<N-1; j++) {
        u(i,j) = h * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + delta_sq * f(i,j));
        diff = (u(i,j) - u_true(i,j));
        d += diff * diff;
      }
    }

    // Find diff
    d = sqrt(d);
    printf("%f\n", d);
    k++;
  }

}
