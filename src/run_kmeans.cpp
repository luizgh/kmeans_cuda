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
#include "wine_data.h"
#include "dataset.h"
#include <algorithm>
#include "kmeans.h"

void printUsageAndExit(const char *msg, char* programName)
{
	if(msg)
		printf("%s\n",msg);
	printf("Usage: %s <paralel|serial> <iris|cifar|wine> <nCentroids> <maxiter>\n", programName);
	exit(1);
}

int main(int argc, char **argv) {

	#ifdef FIXSEED
	std::srand(0);
	#else
	std::srand(std::time(0));
	#endif

	Dataset *dataset;
	Kmeans *kmeans;

	if(argc <= 3 || argc > 5)
		printUsageAndExit("Invalid number of arguments", argv[0]);
	int maxIter = -1;
	if (argc > 4)
		maxIter = atoi(argv[4]);

	int nCentroids = atoi(argv[3]);

	if (strcmp(argv[2], "iris") == 0)
		dataset = new IrisDataset;
	else if (strcmp(argv[2], "cifar") == 0)
		dataset = new Cifar10Dataset_1batch;
	else if (strcmp(argv[2], "wine") == 0)
		dataset = new WineDataset;
	else
		printUsageAndExit("Invalid dataset", argv[0]);

	if (strcmp(argv[1], "serial") == 0)
		kmeans = new KmeansSerial(dataset->getX(), dataset->getNExamples(), dataset->getnDim(), true);
	else if (strcmp(argv[1], "parallel") == 0)
		kmeans = new KmeansParallel(dataset->getX(), dataset->getNExamples(), dataset->getnDim(), true);
	else
		printUsageAndExit("Invalid algorithm", argv[0]);

	kmeans->run(nCentroids, maxIter);

	delete kmeans;
	delete dataset;

	return 0;
}
