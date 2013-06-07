#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "load_iris_data.h"
//#include "load_cifar10_data.h"
#include "kmeans_serial.h"

typedef void (*initFunction)(double *, double *, int);
typedef void (*loadDataFunction)(double **X, int **y);

double *run_kmeans(initFunction InitializeCentroidsFunction,
				   loadDataFunction LoadDataFunction,
				   int nCentroids);

int main(int argc, char **argv) {

	run_kmeans(&InitializeCentroids, load_iris_data, 10);
	/*
	srand(time(NULL));
	if (argc > 1 && strcmp(argv[1], "test") == 0)
	{
		printf("Running test\n\n");
		double *centroidPosition = run_kmeans(&InitializeCentroidsTest, load_iris_data, 3);
		CompareTestResultsAgainstBaseline (centroidPosition);
	}
	else
		

	*/
	return 0;
}


void InitializeCentroids (double *dataX, double *centroidPosition, int nCentroids)
{
	//Initialize centroids with K random examples (Forgy's method)
	int i;
	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i=0; i < nCentroids; i++) {
		selectedExample = rand() % NEXAMPLES;
		printf("%d ", selectedExample);
		centroidPosition[i*NDIM + 0] = dataX[selectedExample*NDIM + 0];
		centroidPosition[i*NDIM + 1] = dataX[selectedExample*NDIM + 1];
		centroidPosition[i*NDIM + 2] = dataX[selectedExample*NDIM + 2];
		centroidPosition[i*NDIM + 3] = dataX[selectedExample*NDIM + 3];
	}
	printf("\n");

}

double *run_kmeans(initFunction InitializeCentroidsFunction,
				   loadDataFunction LoadDataFunction,
				   int nCentroids)
{
	

}