#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

int main(int argc, char **argv) {
  
  // Get number of GPUs
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    printf("cudaGetDeviceCount error %d\n-> %s\n", error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  // Get device propreties and print 
  for (int dev = 0; dev < deviceCount; dev++) {
    struct cudaDeviceProp prop;
    int value;
    printf("\n==========  cudaDeviceGetProperties ============  \n");
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
    printf("\nDevice %d: \"%s\"\n", dev, prop.name);
    printf("  GPU Clock Rate (MHz):                          %d\n", prop.clockRate/1000);
    printf("  Memory Clock Rate (MHz):                       %d\n", prop.memoryClockRate/1000);
    printf("  Memory Bus Width (bits):                       %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s):                  %.2f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  CUDA Cores/MP:                                 %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("  CUDA Cores:                                    %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) *
           prop.multiProcessorCount);
    printf("  Total amount of global memory:                 %.0f GB\n", prop.totalGlobalMem / 1073741824.0f);
    printf("  Total amount of shared memory per block:       %zu kB\n",
           prop.sharedMemPerBlock/1024);
    printf("  Total number of registers available per block: %d\n",
           prop.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           prop.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           prop.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2]);

    printf("\n\n==========  cudaDeviceGetAttribute ============  \n");
    printf("\nDevice %d: \"%s\"\n", dev, prop.name);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxThreadsPerBlock, dev);
    printf("  Max number of threads per block:              %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxBlockDimX, dev);
    printf("  Max block dimension X:                        %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxBlockDimY, dev);
    printf("  Max block dimension Y:                        %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxBlockDimZ, dev);
    printf("  Max block dimension Z:                        %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxGridDimX, dev);
    printf("  Max grid dimension X:                         %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxGridDimY, dev);
    printf("  Max grid dimension Y:                         %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxGridDimZ, dev);
    printf("  Max grid dimension Z:                         %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    printf("  Max shared memory per block:                  %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrWarpSize, dev);
    printf("  Warp size:                                    %d\n",
           value);      
    cudaDeviceGetAttribute (&value, cudaDevAttrClockRate, dev);
    printf("  Peak clock frequency in kilohertz:            %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMemoryClockRate, dev);
    printf("  Peak memory clock frequency in kilohertz:     %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("  Global memory bus width in bits:              %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrL2CacheSize, dev);
    printf("  Size of L2 cache in bytes:                    %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    printf("  Maximum resident threads per SM:              %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrComputeCapabilityMajor, dev);
    printf("  Major compute capability version number:      %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrComputeCapabilityMinor, dev);
    printf("  Minor compute capability version number:      %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    printf("  Max shared memory per SM in bytes:            %d\n",
           value);
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    printf("  Max number of 32-bit registers per SM:        %d\n",
           value);  
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    printf("  Max per block shared mem size on the device:  %d\n",
           value);  
    cudaDeviceGetAttribute (&value, cudaDevAttrMaxBlocksPerMultiprocessor, dev);
    printf("  Max thread blocks that can reside on a SM:    %d\n",
           value);  
  }
}