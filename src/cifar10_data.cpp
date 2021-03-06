#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cifar10_data.h"

Cifar10Dataset_1batch::Cifar10Dataset_1batch(){
	load_cifar_data_from_batches();
}

Cifar10Dataset_1batch::~Cifar10Dataset_1batch(){
	free(X);
	free(y);
}

int Cifar10Dataset_1batch::load_cifar_data_from_batches()
//Expected:
//filename: string pointing to the flat file with iris data
//X: vector of size: sizeof(float) * numExamples * 4
//Y: vector of size: sizeof(int) * numExamples
{
	X = (float*) malloc (sizeof(float) * (nExamples * nDim));
	y = (int*) malloc (sizeof(int) * (nExamples));

	//We only load the first batch due to memory limitations on the GPU
	processBatch(1);
	//TODO process all batches
	/*
	  	int i;
	    for (i=1; i<=5; i++)
		processBatch(i);
	 */
	return 1;
}

void Cifar10Dataset_1batch::processBatch(int batchNumber)
{
	char filename[100];
	sprintf(filename, "data_batch_%d.bin", batchNumber);
	FILE *file = fopen(filename,"r");
	assert(file);

	int i = (batchNumber - 1) * nExamples_perbatch;
	int j;

	unsigned char line[3073];
	while (!feof(file))
	{
		if (!fread(line, 1, 3073, file))
			break;
		//Get line content
		y[i] = line[0];
		for (j = 1; j < 3073; j++)
		{
			X[i * nDim + (j-1)] = (float)line[j];
		}

		i++;
	}
	fclose(file);
}
