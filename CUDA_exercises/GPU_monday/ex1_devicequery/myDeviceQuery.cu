#include <stdio.h>
#include <helper_cuda.h>

int main(int argc, char *argv[])
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int device = 0; device < nDevices; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("Device %i: \"%s\".\n", device, prop.name);
        printf("  Multiprocessors: %i\n", prop.multiProcessorCount);
        printf("  Cores: %i\n",_ConvertSMVer2Cores(prop.major, prop.minor)
               * prop.multiProcessorCount);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Maximum number of threads per block: %d.\n",
               prop.maxThreadsPerBlock);
    }
}
