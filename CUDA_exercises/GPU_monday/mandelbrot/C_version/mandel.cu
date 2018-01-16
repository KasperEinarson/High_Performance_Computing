#include <omp.h>
#include <stdio.h>

#define chunk 20

void
mandel(int disp_width, int disp_height, int *array, int max_iter) {

  double 	scale_real, scale_imag;
  double 	x, y, u, v, u2, v2;
  int 	i, j, iter;

  #pragma omp parallel private(i, j, iter, x, y, u, v, u2, v2) \
    shared(scale_real, scale_imag, disp_width, disp_height, array, max_iter)
  {

  scale_real = 3.5 / (double)disp_width;
  scale_imag = 3.5 / (double)disp_height;

  #pragma omp for schedule(dynamic, chunk)
  for(i = 0; i < disp_width; i++) {

  	x = ((double)i * scale_real) - 2.25;

  	for(j = 0; j < disp_height; j++) {
  	    y = ((double)j * scale_imag) - 1.75;

  	    u    = 0.0;
  	    v    = 0.0;
  	    u2   = 0.0;
  	    v2   = 0.0;
  	    iter = 0;

  	    while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
      		v = 2 * v * u + y;
      		u = u2 - v2 + x;
      		u2 = u*u;
      		v2 = v*v;
      		iter = iter + 1;
  	    }

  	    // if we exceed max_iter, reset to zero
  	    iter = iter == max_iter ? 0 : iter;

  	    array[i*disp_height + j] = iter;
  	}
  }

  } // end omp parallel
}
