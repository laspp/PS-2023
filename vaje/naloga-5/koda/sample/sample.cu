#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1

__global__ void do_nothing(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height)
{
    printf("DID NOTHING\n");
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }
    
    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    cpp = COLOR_CHANNELS;

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    // Kot preizkus samo kopiramo vhodno sliko v izhodno
    memcpy(h_imageOut,h_imageIn,datasize);

    // Nastavimo organizacijo niti v 2D
    dim3 blockSize(1, 1);
    dim3 gridSize(1, 1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Rezervacija pomnilnika na napravi
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // Uporabimo dogodke CUDA za merjenje casa
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Zazenemo scepec
    cudaEventRecord(start);
    do_nothing<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height);
    getLastCudaError("do_nothing() execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // Izpisemo cas
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Zapisemo izhodno sliko v datoteko
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Sprostimo pomnilnik na napravi
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Pocistimo dogodke
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    // Sprostimo pomnilnik na gostitelju
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
