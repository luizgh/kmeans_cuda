/*
 * kmeans_serial.c
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <string>
#include <cassert>
#include <ctime>
#include <cmath>
#include <cfloat>
#include "iris_data.h"
//#include "load_cifar10_data.h"
#include "kmeans.h"
#include "kmeans_serial.h"
#include <algorithm>

long long timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}


KmeansSerial::KmeansSerial(float *data, int nExamples, int nDim,
		bool verbose) {
	this->dataX = data;
	this->nExamples = nExamples;
	this->nDim = nDim;
	this->verbose = verbose;
	this->initializeCentroidsFunction = &InitializeCentroids;
	this->lastRunningTime = 0;
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


float* KmeansSerial::run(int nCentroids, int maxIter) {
	this->lastRunningTime = 0;
	this->nCentroids = nCentroids;
	int iExample, changedFromLastIteration;
	AllocateMemoryForCentroidVariables();

	//InitializeCentroids
	(*initializeCentroidsFunction)(dataX, centroidPosition, nCentroids, nDim, nExamples, verbose);

	 struct timespec start, end;
	  clock_gettime(CLOCK_MONOTONIC, &start);

	changedFromLastIteration = 1;
	int nIteration = 0;
	while (changedFromLastIteration && (nIteration < maxIter || maxIter == -1)) {
		nIteration++;
		if (this->verbose)
			printf("Starting iteration %d\n", nIteration);

		changedFromLastIteration = FindClosestCentroidsAndCheckForChanges();

		//Update centroid location
		ClearIntArray(numberOfExamplePerCentroid, nCentroids);
		ClearfloatArray(runningSumOfExamplesPerCentroid, nCentroids * nDim);

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
	clock_gettime(CLOCK_MONOTONIC, &end);
    long long timeElapsed = timespecDiff(&end, &start);
    lastRunningTime = ((float)timeElapsed)/1000000;

    if (this->verbose)
    {
		printf("done\n");
		printf("Total time: %f ms\n", lastRunningTime);


		printf("Centroids: \n");
		int i;
		for (i = 0; i < nCentroids; i++)
			printf("%.17g %.17g %.17g %.17g\n", centroidPosition[i * nDim + 0],
					centroidPosition[i * nDim + 1], centroidPosition[i * nDim + 2],
					centroidPosition[i * nDim + 3]);

		fflush(stdout);
    }
    FreeCentroidsMemory();
	return centroidPosition;
}


float KmeansSerial::getLastRunningTime() {
	return lastRunningTime;
}

void KmeansSerial::AllocateMemoryForCentroidVariables() {
	//Allocate memory for centroid variables
	centroidPosition = (float*) malloc(sizeof(float) * (nCentroids * nDim));
	centroidAssignedToExample = (int*) malloc(sizeof(int) * nExamples);
	runningSumOfExamplesPerCentroid = (float*) malloc(
			sizeof(float) * (nCentroids * nDim));
	numberOfExamplePerCentroid = (int*) ((malloc(sizeof(int) * nCentroids)));
}

void KmeansSerial::FreeCentroidsMemory() {
	free(centroidPosition);
	free(centroidAssignedToExample);
	free(runningSumOfExamplesPerCentroid);
	free(numberOfExamplePerCentroid);
}

void KmeansSerial::ClearIntArray(int* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0;
}


void KmeansSerial::ClearfloatArray(float* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0.0;
}


void KmeansSerial::InitializeCentroids(float *dataX, float *centroidPosition,
		int nCentroids, int nDim, int nExamples, bool verbose) {
	//Initialize centroids with K random examples (Forgy's method)
    int *randomVector;
    int i,j;

    randomVector = (int*) malloc (sizeof(int) * nExamples);
    for (int i =0; i< nExamples;i++)
    	randomVector[i] = i;

    std::random_shuffle(randomVector, randomVector+ nExamples);

    if (verbose)
    	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i = 0; i < nCentroids; i++) {
		selectedExample = randomVector[i];
		if (verbose)
			printf("%d ", selectedExample);
		for (j = 0; j < nDim; j++)
			centroidPosition[i * nDim + j] = dataX[selectedExample * nDim + j];
	}
	if(verbose)
		printf("\n");

}


float KmeansSerial::CalculateDistance(float *dataX, float *centroidPosition, int iExample,
		int jCentroid) {
	//calculate the distance between a data point and a centroid
	int i;
	float sum = 0;
	float currentVal;
	for (i = 0; i < nDim; i++) {
		currentVal = centroidPosition[jCentroid * nDim + i]
				- dataX[iExample * nDim + i];
		sum += currentVal * currentVal;
	}
	return sqrt(sum);
}


int KmeansSerial::GetClosestCentroid(int iExample) {
	//Find the centroid closest to a data point
	float distanceToCurrentCentroid;
	float smallestDistanceToCentroid = FLT_MAX;
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

