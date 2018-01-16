#include <math.h>
#include <stdlib.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define a(i,l) A[(i)*k + (l)]
#define b(l,j) B[(l)*n + (j)]
#define c(i,j) C[(i)*n + (j)]

// Declarations
void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs);

void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs) {

  int i1, i2, j1, j2, l, i1_min, j1_min;

  for (i1=0; i1<m*n; i1++) C[i1] = 0;
   //printf("%d",m);
   //printf("%d",n);
   //printf("%d",k);
  for (i1 = 0; i1<m; i1+=bs) {
	for (j1 = 0; j1<k; j1+=bs) {
	    
            i1_min = min(i1+bs, m);
	    j1_min = min(j1+bs, k);
            
            for (i2 = i1; i2 < i1_min; ++i2) {
	    	for (j2 = j1; j2 < j1_min; ++j2) {

                    //c(i2,j2) = 0;

                    for (l = 0; l<n; ++l) {
                        c(i2,l) = c(i2,l) + a(i2,j2) * b(j2,l);
                    }
                }
	   }
        }
  }
}
