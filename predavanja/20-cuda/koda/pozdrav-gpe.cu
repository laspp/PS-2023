// prvi program za GPE
// prevajanje: 
//		module load CUDA
//		nvcc -o pozdrav-gpe pozdrav-gpe.cu
// izvajanje: 
//		srun --gpus=1 --partition=gpu ./pozdrav-gpe 2 4

#include <stdio.h>
#include "cuda.h"

__global__ void pozdrav(void) {
	printf("Hello from thread %d.%d!\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char **argv) {

	// preberemo argumente iz ukazne vrstice
	int numBlocks = 0;
	int numThreads = 0;
	if (argc == 3) {
		numBlocks = atoi(argv[1]);
		numThreads = atoi(argv[2]);
	}
	if (numBlocks == 0 || numThreads == 0) {
		printf("usage:\n\t%s <number of blocks> <number of threads>\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// zaženemo kodo na napravi
	dim3 gridSize(numBlocks, 1, 1);
	dim3 blockSize(numThreads, 1, 1);
	pozdrav<<<gridSize, blockSize>>>();
	
	// počakamo, da vse niti na napravi zaključijo
	cudaDeviceSynchronize();
 
	return 0;

}
