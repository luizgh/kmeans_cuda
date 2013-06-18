#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctime>
#include <math.h>
#include <float.h>
#include "iris_data.h"
#include "cifar10_data.h"
#include "kmeans.h"
#include "kmeans_parallel.h"
#include "kmeans_serial.h"
#include <algorithm>

int main(int argc, char **argv) {
    std::srand(std::time(0));

	if(argc <= 2 || argc > 3) 	{
		printf("Usage: %s <iris|cifar> <nCentroids>\n", argv[0]);
		return 1;
	}
	int nCentroids = atoi(argv[2]);

	if (strcmp(argv[1], "iris") == 0)
	{
		IrisDataset d;
		KmeansSerial kmeans (d.X, d.nExamples, d.nDim, true);
		kmeans.run(nCentroids);
	}
	else if (strcmp(argv[1], "cifar") == 0)
	{
		Cifar10Dataset_1batch d;
		KmeansSerial kmeans (d.X, d.nExamples, d.nDim, true);
		kmeans.run(nCentroids);
	}
	else
		printf("Usage: %s <iris|cifar> <nCentroids>\n", argv[0]);

	return 0;
}
