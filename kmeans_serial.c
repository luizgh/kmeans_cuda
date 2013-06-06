/*
 * kmeans_serial.c
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "load_iris_data.h"
#include "kmenas_serial.h"

typedef void (*initFunction)(double *, double *, int);
typedef void (*loadDataFunction)(double **X, int **y);

int main(int argc, char **argv) {
	srand(time(NULL));
	if (argc > 1 && strcmp(argv[1], "test") == 0)
	{
		printf("Running test\n\n");
		double *centroidPosition = run_kmeans(&InitializeCentroidsTest, load_iris_data, 3);
		CompareTestResultsAgainstBaseline (centroidPosition);
	}
	else
		run_kmeans(&InitializeCentroids, load_iris_data, 3);
	return 0;
}

void load_iris_data(double **X, int **y)
{
	load_iris_data_from_file("iris.data", X, y);
}

double *run_kmeans(initFunction InitializeCentroidsFunction,
				   loadDataFunction LoadDataFunction,
				   int nCentroids)
{
	double *dataX;
	int *datay;

	double *centroidPosition;
	int *centroidAssignedToExample;
	double *runningSumOfExamplesPerCentroid;
	int *numberOfExamplePerCentroid;

	int iExample, changedFromLastIteration;

	//Load data
	(*LoadDataFunction)(&dataX, &datay);

	//Allocate memory for centroid variables
	centroidPosition = (double*) malloc(sizeof(double) * (nCentroids * NDIM));
	centroidAssignedToExample = (int*) malloc(sizeof(int) * NEXAMPLES);
	runningSumOfExamplesPerCentroid = (double*) malloc(sizeof(double) * (nCentroids * NDIM));
	numberOfExamplePerCentroid = (int*) malloc(sizeof(int) * nCentroids);

	//InitializeCentroids
	(*InitializeCentroidsFunction)(dataX, centroidPosition, nCentroids);

	changedFromLastIteration = 1;
	int nIteration = 0;

	while (changedFromLastIteration)
	{
		nIteration++;
		printf("Starting iteration %d\n", nIteration);
		//Find closest centroids
		changedFromLastIteration = 0;
		for (iExample = 0; iExample < NEXAMPLES; iExample++) {
			int closestCentroid = GetClosestCentroid(dataX, centroidPosition, centroidAssignedToExample, iExample, nCentroids);
			if (closestCentroid != centroidAssignedToExample[iExample])
				changedFromLastIteration = 1;
			centroidAssignedToExample[iExample] = closestCentroid;
		}

		//Update centroid location
		ClearIntArray(numberOfExamplePerCentroid, nCentroids);
		ClearDoubleArray(runningSumOfExamplesPerCentroid, nCentroids * NDIM);

		int currentCentroid;
		for (iExample = 0; iExample < NEXAMPLES; iExample++) {
			currentCentroid = centroidAssignedToExample[iExample];
			numberOfExamplePerCentroid[currentCentroid]++;
			int jDim;
			for (jDim=0; jDim< NDIM; jDim++)
				runningSumOfExamplesPerCentroid[currentCentroid * NDIM + jDim] += dataX[iExample * NDIM + jDim];
		}
		for (currentCentroid = 0; currentCentroid < nCentroids; currentCentroid++)
		{
			int jDim;
			for (jDim=0; jDim< NDIM; jDim++)
				centroidPosition[currentCentroid*NDIM + jDim] = runningSumOfExamplesPerCentroid[currentCentroid * NDIM + jDim] /
																numberOfExamplePerCentroid[currentCentroid];
		}

	}

	printf("done\n");
	printf("Centroids: \n");
	int i;
	for (i = 0; i < nCentroids; i++)
		printf("%.17g %.17g %.17g %.17g\n", centroidPosition[i * NDIM + 0], centroidPosition[i * NDIM + 1], centroidPosition[i * NDIM + 2], centroidPosition[i * NDIM + 3]);
	fflush(stdout);
	return centroidPosition;
}


void ClearIntArray(int *vector, int size)

{
	int i;
	for (i=0; i<size; i++)
		vector[i] = 0;
}

void ClearDoubleArray(double *vector, int size)
{
	int i;
	for (i=0; i<size; i++)
		vector[i] = 0.0;
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

double CalculateDistance(double *dataX, double *centroidPosition, int iExample, int jCentroid)
{
	//calculate the distance between a data point and a centroid
	int i;
	double sum = 0;
	double currentVal;
	for (i=0; i<NDIM; i++)
	{
		currentVal = centroidPosition[jCentroid*NDIM + i] - dataX[iExample*NDIM + i];
		sum += currentVal * currentVal;
	}
	return sqrt(sum);
}

int GetClosestCentroid (double *dataX, double *centroidPosition, int *centroidAssignedToExample, int iExample, int nCentroids)
{
	//Find the centroid closest to a data point
	double distanceToCurrentCentroid;
	double smallestDistanceToCentroid = DBL_MAX;
	int assignedCentroid = -1;
	int jCentroid;

	for (jCentroid = 0; jCentroid < nCentroids; jCentroid++) {
		distanceToCurrentCentroid = CalculateDistance(dataX, centroidPosition, iExample, jCentroid);
		if (distanceToCurrentCentroid < smallestDistanceToCentroid)
		{
			smallestDistanceToCentroid = distanceToCurrentCentroid;
			assignedCentroid = jCentroid;
		}
	}
	assert (assignedCentroid != -1);
	return assignedCentroid;
}



void InitializeCentroidsTest (double *dataX, double *centroidPosition, int nCentroids)
{
	//Initialize centroids with K random examples (Forgy's method)
	int i;
	int selectedExamples[] = {16 , 85, 79};
	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i=0; i<nCentroids; i++) {
		selectedExample = selectedExamples[i];
		printf("%d ", selectedExample);
		centroidPosition[i*NDIM + 0] = dataX[selectedExample*NDIM + 0];
		centroidPosition[i*NDIM + 1] = dataX[selectedExample*NDIM + 1];
		centroidPosition[i*NDIM + 2] = dataX[selectedExample*NDIM + 2];
		centroidPosition[i*NDIM + 3] = dataX[selectedExample*NDIM + 3];
	}
	printf("\n");

}

void CompareTestResultsAgainstBaseline (double *centroidPosition)
{
	int nCentroids = 3;
	double baseline[] = {5.0059999999999993, 3.4180000000000006, 1.464, 0.24399999999999991,
			6.8538461538461526, 3.0769230769230766, 5.7153846153846146, 2.0538461538461532,
			5.8836065573770497, 2.7409836065573772, 4.3885245901639349, 1.4344262295081966};
	int i;
	double maxError = 1e-5;
	double error = 0;
	for (i=0; i< nCentroids * NDIM; i++)
		error += fabs(centroidPosition[i] - baseline[i]);

	assert (error < maxError);
	printf ("OK!! Error agains baseline below threshold: %lf\n", error);
}
