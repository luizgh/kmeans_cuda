/*
 * kmeans_parallel.c
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
#include "utils.h"
#include "kmeans.h"
#include "kmeans_parallel.h"


__global__
void run_kmeans_parallel(float *d_dataX, float *d_centroidPosition,
		int *d_centroidAssignedToExample,
		float *d_runningSumOfExamplesPerCentroid,
		int *d_numberOfExamplePerCentroid, int nExamples, int nCentroids, int nDim, int *d_changedSinceLastIteration) {

	int myExample = blockIdx.x * blockDim.x + threadIdx.x;
	if (myExample >= nExamples)
		return; //out of range

	int i;
	float sum = 0;
	float currentVal;

	//Find closest centroid to current example
	float distanceToCurrentCentroid;
	float smallestDistanceToCentroid = FLT_MAX;
	int assignedCentroid = -1;
	int jCentroid;
	for (jCentroid = 0; jCentroid < nCentroids; jCentroid++) {
		sum = 0;

		for (i = 0; i < nDim; i++) {
			currentVal = d_centroidPosition[jCentroid * nDim + i]
					- d_dataX[myExample * nDim + i];
			sum += currentVal * currentVal;
		}
		distanceToCurrentCentroid = sqrt(sum);

		if (distanceToCurrentCentroid < smallestDistanceToCentroid) {
			smallestDistanceToCentroid = distanceToCurrentCentroid;
			assignedCentroid = jCentroid;
		}
	}
	if (d_centroidAssignedToExample[myExample] != assignedCentroid)
		*d_changedSinceLastIteration = 1;

	d_centroidAssignedToExample[myExample] = assignedCentroid;

	atomicAdd(&d_numberOfExamplePerCentroid[assignedCentroid], 1);
	for (i = 0; i < nDim; i++) {
		atomicAdd(&d_runningSumOfExamplesPerCentroid[assignedCentroid], d_dataX[myExample * nDim + i]);
	}

	//synchronize threads
	__syncthreads();

	if (myExample < nCentroids)
	{
		int jDim;
		for (jDim = 0; jDim < nDim; jDim++)
			d_centroidPosition[myExample * nDim + jDim] =
					d_runningSumOfExamplesPerCentroid[myExample * nDim
							+ jDim]
							/ d_numberOfExamplePerCentroid[myExample];
	}



}

/*
	changedFromLastIteration = 1;
	int nIteration = 0;
	while (changedFromLastIteration) {
		nIteration++;
		if (this->verbose)
			printf("Starting iteration %d\n", nIteration);


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

	}*/


KmeansParallel::KmeansParallel(float *data, int nExamples, int nDim,
		bool verbose) {
	this->dataX = data;
	this->nExamples = nExamples;
	this->nDim = nDim;
	this->verbose = verbose;
	this->initializeCentroidsFunction = &InitializeCentroids;
}

void KmeansParallel::setInitializeCentroidsFunction(initFunction fun) {
	initializeCentroidsFunction = fun;
}

int KmeansParallel::FindClosestCentroidsAndCheckForChanges() {
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

float* KmeansParallel::run(int nCentroids) {
	this->nCentroids = nCentroids;
	AllocateMemoryForCentroidVariables();

	//InitializeCentroids
	(*initializeCentroidsFunction)(dataX, centroidPosition, nCentroids, nDim,
			nExamples);

	AllocateMemoryAndCopyVariablesToGPU();

	//1 thread per example
	int blockSize_1d = 512;
	int gridSize_1d = nExamples / blockSize_1d + 1;

	printf ("nExamples: %d; blockSize: %d; gridSize:%d\n", nExamples, blockSize_1d, gridSize_1d);

    const dim3 blockSize (blockSize_1d, 1, 1);
    const dim3 gridSize(gridSize_1d, 1, 1);



	run_kmeans_parallel<<<gridSize, blockSize>>> (d_dataX, d_centroidPosition,
			d_centroidAssignedToExample, d_runningSumOfExamplesPerCentroid, d_numberOfExamplePerCentroid, nExamples, nCentroids, nDim, d_changedSinceLastIteration);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	ClearfloatArray(centroidPosition, nCentroids * nDim);

	CopyResultsFromGPU();

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

void KmeansParallel::AllocateMemoryForCentroidVariables() {
	//Allocate memory for centroid variables
	centroidPosition = (float*) malloc(sizeof(float) * (nCentroids * nDim));
	centroidAssignedToExample = (int*) malloc(sizeof(int) * nExamples);
	runningSumOfExamplesPerCentroid = (float*) malloc(
			sizeof(float) * (nCentroids * nDim));
	numberOfExamplePerCentroid = (int*) ((malloc(sizeof(int) * nCentroids)));
}

void KmeansParallel::AllocateMemoryAndCopyVariablesToGPU() {
	checkCudaErrors(cudaMalloc(&d_dataX, sizeof(float) * nExamples * nDim ));
	checkCudaErrors(
			cudaMalloc(&d_centroidPosition, sizeof(float) * (nCentroids * nDim)));
	checkCudaErrors(
			cudaMalloc(&d_centroidAssignedToExample, sizeof(int) * nExamples));
	checkCudaErrors(
			cudaMalloc(&d_runningSumOfExamplesPerCentroid, sizeof(float) * (nCentroids * nDim)));
	checkCudaErrors(
			cudaMalloc(&d_numberOfExamplePerCentroid, sizeof(int) * nCentroids ));
	checkCudaErrors(
				cudaMalloc(&d_changedSinceLastIteration, sizeof(int) ));

	checkCudaErrors(
			cudaMemcpy(d_centroidPosition, centroidPosition, sizeof(float) * (nCentroids * nDim), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_dataX, dataX, sizeof(float) * (nExamples * nDim), cudaMemcpyHostToDevice));
}

void KmeansParallel::CopyResultsFromGPU(){
	checkCudaErrors(cudaMemcpy(centroidPosition, d_centroidPosition, sizeof(float) * (nCentroids * nDim), cudaMemcpyDeviceToHost));
}

void KmeansParallel::ClearIntArray(int* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0;
}

void KmeansParallel::ClearfloatArray(float* vector, int size) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = 0.0;
}

void KmeansParallel::InitializeCentroids(float *dataX, float *centroidPosition,
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

float KmeansParallel::CalculateDistance(float *dataX, float *centroidPosition,
		int iExample, int jCentroid) {
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

int KmeansParallel::GetClosestCentroid(int iExample) {
	//Find the centroid closest to a data point
	float distanceToCurrentCentroid;
	float smallestDistanceToCentroid = DBL_MAX;
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

void KmeansParallel::CompareTestResultsAgainstBaseline(
		float *centroidPosition) {
	int nCentroids = 3;
	float baseline[] = { 5.0059999999999993, 3.4180000000000006, 1.464,
			0.24399999999999991, 6.8538461538461526, 3.0769230769230766,
			5.7153846153846146, 2.0538461538461532, 5.8836065573770497,
			2.7409836065573772, 4.3885245901639349, 1.4344262295081966 };
	int i;
	float maxError = 1e-5;
	float error = 0;
	for (i = 0; i < nCentroids * nDim; i++)
		error += fabs(centroidPosition[i] - baseline[i]);

	assert(error < maxError);
	printf("OK!! Error agains baseline below threshold: %lf\n", error);
}

