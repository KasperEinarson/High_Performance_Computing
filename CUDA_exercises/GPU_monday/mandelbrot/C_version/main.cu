#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "mandelgpu.h"
#include "writepng.h"

int
main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int   *h_image, *d_image;

    width    = 2601;
    height   = 2601;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    //image = (int *)malloc( width * height * sizeof(int));
    cudaMallocHost((void **)&h_image, width * height * sizeof(int));
    cudaMalloc((void **)&d_image, width * height * sizeof(int));

    dim3 dimGrid(height/16, width/16, 1);
    dim3 dimBlock(16, 16, 1);

    mandelgpu<<<dimGrid,dimBlock>>>(width, height, d_image, max_iter);
    cudaDeviceSynchronize();

    cudaMemcpy(h_image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    writepng("mandelbrot.png", h_image, width, height);

    cudaFreeHost(h_image);
    cudaFree(d_image);

    return(0);
}
