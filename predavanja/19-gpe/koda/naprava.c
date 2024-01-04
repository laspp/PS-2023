// informacije o napravi
//  funkciji _ConvertSMVer2Cores in _ConvertSMVer2ArchName sta iz https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
//  prevajanje:
//      module load CUDA
//      nvcc -o naprava naprava.c
// izvajanje:
//      srun --partition=gpu --gpus=1 ./naprava

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
                // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
        return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

const char *_ConvertSMVer2ArchName(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the GPU Arch name)
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        const char *name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"},
        {0x32, "Kepler"},
        {0x35, "Kepler"},
        {0x37, "Kepler"},
        {0x50, "Maxwell"},
        {0x52, "Maxwell"},
        {0x53, "Maxwell"},
        {0x60, "Pascal"},
        {0x61, "Pascal"},
        {0x62, "Pascal"},
        {0x70, "Volta"},
        {0x72, "Xavier"},
        {0x75, "Turing"},
        {0x80, "Ampere"},
        {0x86, "Ampere"},
        {0x87, "Ampere"},
        {0x89, "Ada"},
        {0x90, "Hopper"},
        {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1)
    {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor))
        {
        return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoArchName for SM %d.%d is undefined."
        "  Default to use %s\n",
        major, minor, nGpuArchNameSM[index - 1].name);
    return nGpuArchNameSM[index - 1].name;
}

int main(int argc, char **argv) 
{
    
    // Get number of GPUs
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) 
    {
        printf("cudaGetDeviceCount error %d\n-> %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Get propreties of each device
    for (int dev = 0; dev < deviceCount; dev++) 
    {

        struct cudaDeviceProp prop;
        int value;

        cudaGetDeviceProperties(&prop, dev);
        
        printf("\n======= Device %d: \"%s\" =======\n", dev, prop.name);
        printf("\ncudaDeviceGetProperties:\n");
        printf("  CUDA Architecture:                             %s, %d.%d\n", _ConvertSMVer2ArchName(prop.major, prop.minor), prop.major, prop.minor);
        printf("\n");
        printf("  GPU clock rate (MHz):                          %d\n", prop.clockRate/1000);
        printf("  Memory clock rate (MHz):                       %d\n", prop.memoryClockRate/1000);
        printf("  Memory bus width (bits):                       %d\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth (GB/s):                  %.0f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("\n");
        printf("  Number of MPs:                                 %d\n", prop.multiProcessorCount);
        printf("  Number of cores per MP:                        %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
        printf("  Total number of cores:                         %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        printf("\n");
        printf("  Total amount of global memory (GB):            %.0f\n", prop.totalGlobalMem / 1073741824.0f);
        printf("  Total amount of shared memory per MP (kB):     %d\n", prop.sharedMemPerMultiprocessor/1024);
        printf("  Total amount of shared memory per block (kB):  %zu\n", prop.sharedMemPerBlock/1024);
        printf("  Maximum number of registers per MP:            %d\n", prop.regsPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("\n");
        printf("  Maximum number of threads per MP:              %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("\n");
        printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        printf("\ncudaDeviceGetAttribute:\n");
        cudaDeviceGetAttribute (&value, cudaDevAttrL2CacheSize, dev);
        printf("  Size of L2 cache in MB:                        %.0f\n", value/1048576.0f);
        cudaDeviceGetAttribute (&value, cudaDevAttrMaxBlocksPerMultiprocessor, dev);  // works in newer CUDA versions
        printf("  Maximum nuber of blocks per MP:                %d\n", value);  
    }
}
