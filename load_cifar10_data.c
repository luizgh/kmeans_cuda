#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "load_cifar10_data.h"



/*load iris data into a matrix X, and vector y. Sample data format:
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
*/

int load_cifar_data_from_batches(double **X, int **y)
//Expected:
//filename: string pointing to the flat file with iris data
//X: vector of size: sizeof(double) * numExamples * 4
//Y: vector of size: sizeof(int) * numExamples
{
	*X = (double*) malloc (sizeof(double) * (NEXAMPLES * NDIM));
	*y = (int*) malloc (sizeof(int) * (NEXAMPLES));

	int i;
	for (i=1; i<=5; i++)
		processBatch(i, X, y);

	printf("%lf %d\n", (*X)[(NEXAMPLES*NDIM)-1], (*y)[NEXAMPLES-1]);
	return 1;
}

void processBatch(int batchNumber,  double **X, int **y)
{
	char filename[100];
	sprintf(filename, "data_batch_%d.bin", batchNumber);
	FILE *file = fopen(filename,"r");
	assert(file != NULL);

	int i = (batchNumber - 1) * NEXAMPLES_PER_BATCH;
	int j;

	unsigned char line[3073];
	while (!feof(file))
	{
		if (!fread(line, 1, 3073, file))
			break;
		//Get line content
		(*y)[i] = line[0];
		for (j = 1; j < 3073; j++)
		{
			(*X)[i * NDIM + (j-1)] = (double)line[j];
		}

		i++;
	}
	fclose(file);
}
