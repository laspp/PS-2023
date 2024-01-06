// računanje razlike vektorjev
// 		argument podamo število blokov, število niti in dolžino vektorjev
// 		elementi vektorjev so inicializirani naključno
// slaba rešitev: ne deluje, če je elementov več kot niti

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

__global__ void vectorSubtract(float *c, const float *a, const float *b, int len) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;	
    if (gid < len)
	    c[gid] = a[gid] - b[gid];
}

void check4error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("napaka: %d (%s)\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
	// preberemo argumente iz ukazne vrstice
	int numBlocks = 0;
	int numThreads = 0;
	int vectorLength = 0;
	if (argc == 4) {
		numBlocks = atoi(argv[1]);
		numThreads = atoi(argv[2]);
		vectorLength = atoi(argv[3]);
	}
	if (numBlocks <= 0 || numThreads <= 0 || vectorLength <= 0) {
		printf("usage:\n\t%s <number of blocks> <number of threads> <vector length>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// rezerviramo pomnilnik na gostitelju
	float *hc = (float *)malloc(vectorLength * sizeof(float));
	float *ha = (float *)malloc(vectorLength * sizeof(float));
	float *hb = (float *)malloc(vectorLength * sizeof(float));

	// rezerviramo pomnilnik na napravi
	float *da, *db, *dc;
	check4error(cudaMalloc((void**)&da, vectorLength * sizeof(float)));
	check4error(cudaMalloc((void**)&db, vectorLength * sizeof(float)));
	check4error(cudaMalloc((void**)&dc, vectorLength * sizeof(float)));

	// nastavimo vrednosti vektorjev a in b na gostitelju
	srand(time(NULL));
	for (int i = 0; i < vectorLength; i++) {
		ha[i] = (float)rand()/RAND_MAX;
		hb[i] = (float)rand()/RAND_MAX;
	}

	// prenesemo vektorja a in b iz gostitelja na napravo
	check4error(cudaMemcpy(da, ha, vectorLength * sizeof(float), cudaMemcpyHostToDevice));
	check4error(cudaMemcpy(db, hb, vectorLength * sizeof(float), cudaMemcpyHostToDevice));

	// zaženemo kodo na napravi
	dim3 gridSize(numBlocks, 1, 1);
	dim3 blockSize(numThreads, 1, 1);
	vectorSubtract<<<gridSize, blockSize>>>(dc, da, db, vectorLength);
	check4error(cudaGetLastError());

	// počakamo, da vse niti na napravi zaključijo
	check4error(cudaDeviceSynchronize());

	// vektor c prekopiramo iz naprave na gostitelja
	check4error(cudaMemcpy(hc, dc, vectorLength * sizeof(float), cudaMemcpyDeviceToHost));

	// preverimo rezultat
	int ok = 1;
	for (int i = 0; i < vectorLength; i++)
		ok &= (ha[i] - hb[i]) == hc[i];
	printf("Result is %s.\n", ok == 1 ? "correct": "wrong");

	// sprostimo pomnilnik na napravi
	check4error(cudaFree(dc));
	check4error(cudaFree(da));
	check4error(cudaFree(db));

	// sprostimo pomnilnik na gostitelju
	free(hc);
	free(ha);
	free(hb);

	return 0;
}
