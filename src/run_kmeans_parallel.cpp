#include <stdio.h>
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

	#ifdef NDEBUG
	std::srand(0);
	#else
	std::srand(std::time(0));
	#endif


	if(argc <= 2 || argc > 4) 	{
		printf("Usage: %s <iris|cifar> <nCentroids> <maxiter>\n", argv[0]);
		return 1;
	}
	int maxIter = -1;
	if (argc >3)
		maxIter = atoi(argv[3]);

	int nCentroids = atoi(argv[2]);

	if (strcmp(argv[1], "iris") == 0)
	{
		IrisDataset d;
		KmeansParallel kmeans (d.X, d.nExamples, d.nDim, true);
		kmeans.run(nCentroids, maxIter);
	}
	else if (strcmp(argv[1], "cifar") == 0)
	{
		Cifar10Dataset_1batch d;
		KmeansParallel kmeans (d.X, d.nExamples, d.nDim, true);
		kmeans.run(nCentroids, maxIter);
	}
	else
		printf("Usage: %s <iris|cifar> <nCentroids>\n", argv[0]);

	return 0;
}
