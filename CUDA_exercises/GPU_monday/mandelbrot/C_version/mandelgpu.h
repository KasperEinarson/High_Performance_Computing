#ifndef __MANDELGPU_H
#define __MANDELGPU_H

__global__ void mandelgpu(int width, int height, int *image, int max_iter);

#endif
