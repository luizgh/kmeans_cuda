#include <stdio.h>
#include <stdlib.h>
#include "wine_data.h"
#include <string.h>
#include <cassert>


WineDataset::WineDataset() {
	int result = load_wine_data_from_file("winequality-red.csv", "winequality-white.csv");
	assert(result);
}

WineDataset::~WineDataset() {
	free(X);
	free(y);
}

int WineDataset::load_wine_data_from_file(const char *filename_red, const char *filename_white)

{
	X = (float*) malloc (sizeof(float) * (nExamples * nDim));
	y = (int*) malloc (sizeof(int) * (nExamples));


	index=0;
	if (!read_file(filename_red)) return 0;
	if (!read_file(filename_white)) return 0;

	return 1;
}

int WineDataset::read_file(const char *filename)
{
	FILE *file = fopen(filename,"r");

	if (!file) return 0;

	char line[1000];
	float tempx[12];
	int j = 0;

	fgets(line, sizeof(line), file); //ignore first line
	while (fgets(line, sizeof(line), file))
	{

		//Get line content
		sscanf(line, "%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f", &tempx[0], &tempx[1], &tempx[2], &tempx[3],
				&tempx[4], &tempx[5], &tempx[6], &tempx[7], &tempx[8], &tempx[9], &tempx[10], &tempx[11]);

		//Assign X
		X[index*nDim + 0] = tempx[0];
		X[index*nDim + 1] = tempx[1];
		X[index*nDim + 2] = tempx[2];
		X[index*nDim + 3] = tempx[3];
		X[index*nDim + 4] = tempx[4];
		X[index*nDim + 5] = tempx[5];
		X[index*nDim + 6] = tempx[6];
		X[index*nDim + 7] = tempx[7];
		X[index*nDim+ 8] = tempx[8];
		X[index*nDim + 9] = tempx[9];
		X[index*nDim + 10] = tempx[10];

		y[index] = tempx[11];
		index++;
	}
	fclose(file);
	return 1;
}
