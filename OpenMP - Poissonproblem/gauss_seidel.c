#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "gauss_seidel.h"

#define u(i,j) U[(i)*N + (j)]
#define f(i,j) F[(i)*N + (j)]

void gauss_seidel(int N, int max_it, double tol, double *U, int *F) {

  clock_t start, end;
  double cpu_time_used, iter_per_sec;
  start= clock();

  int i, j, k = 0;
  double h = 1.0 / 4.0;
  double delta = 2/((double)N - 1.0), delta_sq = delta * delta;
  double d = tol + 1.0;
  double prev = 0.0;
  double diff;

  // Initialize U
  for (i=0; i<N; i++) {
    u(i,0) = 20.0;
    u(0,i) = 20.0;
    u(i,N-1) = 20.0;
  }
  for (i=1; i<N; i++) {
    for (j=1; j<N-1; j++) {
      u(i,j) = 0.0;
    }
  }

  while (d > tol && k < max_it) {
    d = 0.0;

    // Update U
    for (i=1; i<N-1; i++) {
      for (j=1; j<N-1; j++) {
        prev = u(i,j);
        u(i,j) = h * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + delta_sq * (double)f(i,j));
        diff = (u(i,j) - prev);
        d += diff * diff;
      }
    }

    // Find diff
    d = sqrt(d);
    k++;

    end = clock();
    cpu_time_used = (double) (end - start) / CLOCKS_PER_SEC;
    iter_per_sec = (double)k / cpu_time_used;
    printf("iter per sec: %f\n", iter_per_sec);
  }

}
