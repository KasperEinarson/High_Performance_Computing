#include <math.h>
#define min(x,y) (((x) < (y)) ? (x) : (y))
#include <stdlib.h>

#define a(i,l) A[(i)*k + (l)]
#define b(l,j) B[(l)*n + (j)]
#define c(i,j) C[(i)*n + (j)]

// Declarations
void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs);

void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs) {

  int i1, i2, j1, j2, l;

  for (i1 = 0; i1<m; i1+=bs) {

  	for (i2 = i1; i2 < min(m, i1+bs); i2++) {

	    for (j1 = 0; j1<n; j1+=bs) {

	    	for (j2 = j1; j2 < min(n, j1+bs); j2++) {

          c(i2,j2) = 0;

          for (l = 0; l<k; l++) {

            c(i2,j2) = c(i2,j2) + a(i2,l) * b(l,j2);

          }
        }
	    }
    }
  }
}
