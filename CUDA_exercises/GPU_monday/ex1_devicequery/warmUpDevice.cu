#include <omp.h>
#include <stdio.h>

const int device = 0;

int main(int argc, char *argv[])
{
    // Wake up GPU from power save state.
    printf("Warming up device %i ... ", device); fflush(stdout);
    double time = omp_get_wtime();
    cudaSetDevice(device);           // Set the device to 0 or 1.
    double *dummy_d;
    cudaMalloc((void**)&dummy_d, 0); // We force the creation of context on the
                                     // device by allocating a dummy variable.
    printf("time = %3.2f seconds\n", omp_get_wtime() - time);
}
