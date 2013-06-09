/*
 * kmeans_serial.c
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <ctime>
#include <cmath>
#include <cfloat>
#include "iris_data.h"
//#include "load_cifar10_data.h"
#include "kmeans.h"
#include "kmeans_serial.h"

KmeansSerial::KmeansSerial(double *data, int nExamples, int nDim,
		bool verbose) {
	this->dataX = data;
	this->nExamples = nExamples;
	this->nDim = nDim;
	this->verbose = verbose;
	this->initializeCentroidsFunction = &InitializeCentroids;
}

void KmeansSerial::setInitializeCentroidsFunction(initFunction fun) {
	initializeCentroidsFunction = fun;
}

int KmeansSerial::FindClosestCentroidsAndCheckForChanges() {
	//Find closest centroids
	int changedFromLastIteration = 0;
	for (int iExample = 0; iExample < nExamples; iExample++) {
		int closestCentroid = GetClosestCentroid(iExample);
		if (closestCentroid != centroidAssignedToExample[iExample])
			changedFromLastIteration = 1;
		centroidAssignedToExample[iExample] = closestCentroid;
	}
	return changedFromLastIteration;
}


double* KmeansSerial::run(int nCentroids) {
	this->nCentroids = nCentroids;
	int iExample, changedFromLastIteration;
	AllocateMemoryForCentroidVariables();

	//InitializeCentroids
	(*initializeCentroidsFunction)(dataX, centroidPosition, nCentroids, nDim, nExamples);

	changedFromLastIteration = 1;
	int nIteration = 0;
	while (changedFromLastIteration) {
		nIteration++;
		if (this->verbose)
			printf("Starting iteration %d\n", nIteration);

		changedFromLastIteration = FindClosestCentroidsAndCheckForChanges();

		//Update centroid location
		ClearIntArray(numberOfExamplePerCentroid, nCentroids);
		ClearDoubleArray(runningSumOfExamplesPerCentroid, nCentroids * nDim);

		int currentCentroid;
		for (iExample = 0; iExample < nExamples; iExample++) {
			currentCentroid = centroidAssignedToExample[iExample];
			numberOfExamplePerCentroid[currentCentroid]++;
			int jDim;
			for (jDim = 0; jDim < nDim; jDim++)
				runningSumOfExamplesPerCentroid[currentCentroid * nDim + jDim] +=
						dataX[iExample * nDim + jDim];
		}
		for (currentCentroid = 0; currentCentroid < nCentroids;
				currentCentroid++) {
			int jDim;
			for (jDim = 0; jDim < nDim; jDim++)
				centroidPosition[currentCentroid * nDim + jDim] =
						runningSumOfExamplesPerCentroid[currentCentroid * nDim
								+ jDim]
								/ numberOfExamplePerCentroid[currentCentroid];
		}

	}
	printf("done\n");
	printf("Centroids: \n");
	int i;
	for (i = 0; i < nCentroids; i++)
		printf("%.17g %.17g %.17g %.17g\n", centroidPosition[i * nDim + 0],
				centroidPosition[i * nDim + 1], centroidPosition[i * nDim + 2],
				centroidPosition[i * nDim + 3]);
	fflush(stdout);
	return centroidPosition;
}


void KmeansSerial::AllocateMemoryForCentroidVariables() {
	//Allocate memory for centroid variables
	centroidPosition = (double*) malloc(sizeof(double) * (nCentroids * nDim));
	centroidAssignedToExample = (int*) malloc(sizeof(int) * nExamples);
	runningSumOfExamplesPerCentroid = (double*) malloc(
			sizeof(double) * (nCentroids * nDim));
	numberOfExamplePerCentroid = (int*) ((malloc(sizeof(int) * nCentroids)));
}


void KmeansSerial::ClearIntArray(int* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0;
}


void KmeansSerial::ClearDoubleArray(double* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0.0;
}


void KmeansSerial::InitializeCentroids(double *dataX, double *centroidPosition,
		int nCentroids, int nDim, int nExamples) {
	//Initialize centroids with K random examples (Forgy's method)
	int i;
	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i = 0; i < nCentroids; i++) {
		selectedExample = rand() % nExamples;
		printf("%d ", selectedExample);
		centroidPosition[i * nDim + 0] = dataX[selectedExample * nDim + 0];
		centroidPosition[i * nDim + 1] = dataX[selectedExample * nDim + 1];
		centroidPosition[i * nDim + 2] = dataX[selectedExample * nDim + 2];
		centroidPosition[i * nDim + 3] = dataX[selectedExample * nDim + 3];
	}
	printf("\n");

}


double KmeansSerial::CalculateDistance(double *dataX, double *centroidPosition, int iExample,
		int jCentroid) {
	//calculate the distance between a data point and a centroid
	int i;
	double sum = 0;
	double currentVal;
	for (i = 0; i < nDim; i++) {
		currentVal = centroidPosition[jCentroid * nDim + i]
				- dataX[iExample * nDim + i];
		sum += currentVal * currentVal;
	}
	return sqrt(sum);
}


int KmeansSerial::GetClosestCentroid(int iExample) {
	//Find the centroid closest to a data point
	double distanceToCurrentCentroid;
	double smallestDistanceToCentroid = DBL_MAX;
	int assignedCentroid = -1;
	int jCentroid;
	for (jCentroid = 0; jCentroid < nCentroids; jCentroid++) {
		distanceToCurrentCentroid = CalculateDistance(dataX, centroidPosition,
				iExample, jCentroid);
		if (distanceToCurrentCentroid < smallestDistanceToCentroid) {
			smallestDistanceToCentroid = distanceToCurrentCentroid;
			assignedCentroid = jCentroid;
		}
	}

	assert(assignedCentroid != -1);
	return assignedCentroid;
}


void KmeansSerial::CompareTestResultsAgainstBaseline(double *centroidPosition) {
	int nCentroids = 3;
	double baseline[] = { 5.0059999999999993, 3.4180000000000006, 1.464,
			0.24399999999999991, 6.8538461538461526, 3.0769230769230766,
			5.7153846153846146, 2.0538461538461532, 5.8836065573770497,
			2.7409836065573772, 4.3885245901639349, 1.4344262295081966 };
	int i;
	double maxError = 1e-5;
	double error = 0;
	for (i = 0; i < nCentroids * nDim; i++)
		error += fabs(centroidPosition[i] - baseline[i]);

	assert(error < maxError);
	printf("OK!! Error agains baseline below threshold: %lf\n", error);
}


