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
#include <algorithm>
#include "cudaTimer.h"


__global__
void clear_vectors(float *d_runningSumOfExamplesPerCentroid,
		int *d_numberOfExamplePerCentroid, int nCentroids, int nDim, int *d_changedSinceLastIteration )
{
	int myExample = blockIdx.x * blockDim.x + threadIdx.x;

		if (myExample < nCentroids)
		{
			for (int i = 0; i < nDim; i++)
				d_runningSumOfExamplesPerCentroid[myExample * nDim + i] = 0;
			d_numberOfExamplePerCentroid[myExample] = 0;
		}
		*d_changedSinceLastIteration = 0;
}

#define BLOCKSIZEEXAMPLES 30
#define BLOCKSIZECENTROIDS 30
#define DIMENSIONS 11
__global__
void run_kmeans_parallel(float *d_dataX, float *d_centroidPosition,
		int *d_centroidAssignedToExample,
		float *d_runningSumOfExamplesPerCentroid,
		int *d_numberOfExamplePerCentroid, int nExamples, int nCentroids, int nDim, float *d_distanceExampleCentroid) {


	int myExample = blockIdx.x * blockDim.x + threadIdx.x;
	int myCentroid = blockIdx.y * blockDim.y + threadIdx.y;
	if (myExample >= nExamples || myCentroid >= nCentroids)
		return; //out of range
    int i;
//Copy data to shared memory
#ifdef USESHAREDMEMORY
	__shared__ float examples[BLOCKSIZEEXAMPLES][DIMENSIONS];
    __shared__ float centroids[BLOCKSIZECENTROIDS][DIMENSIONS];


    //Note: Only works if all blocks have more than nDim threads in each dimension (x.y)
    if (threadIdx.y < nDim)
    	examples[threadIdx.x][threadIdx.y] = d_dataX[myExample * nDim + threadIdx.y];

    if (threadIdx.x < nDim)
    	centroids[threadIdx.y][threadIdx.x] = d_centroidPosition [myCentroid * nDim + threadIdx.x];

    __syncthreads();
#endif
	float sum = 0;
	float currentVal;

	//Find closest centroid to current example
	float distanceToCurrentCentroid;

	sum = 0;

	for (i = 0; i < nDim; i++) {

#ifdef USESHAREDMEMORY
		//printf("BlockIDx: %d, BlockIDy: %d, ThreadIDx: %d, ThreadIDy: %d; Examples: %f %f; Centroids: %f, %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, examples[threadIdx.x][i], d_dataX[myExample * nDim + i], centroids[threadIdx.y][i], d_centroidPosition[myCentroid * nDim + i]);
//		assert(centroids[threadIdx.y][i] == d_centroidPosition[myCentroid * nDim + i]);
//		assert(examples[threadIdx.x][i] == d_dataX[myExample * nDim + i]);


		currentVal = centroids[threadIdx.y][i] - examples[threadIdx.x][i];
#else
		currentVal = d_centroidPosition[myCentroid * nDim + i] - d_dataX[myExample * nDim + i];
#endif

		sum += currentVal * currentVal;
	}
	distanceToCurrentCentroid = sqrt(sum);

	d_distanceExampleCentroid[myExample * nCentroids + myCentroid] = distanceToCurrentCentroid;

}

__global__
void aggregate_centroid_locations(float *d_runningSumOfExamplesPerCentroid,
		int *d_numberOfExamplePerCentroid, int *d_centroidAssignedToExample, float *d_dataX, int nDim, int nExamples, int nCentroids, float *d_distanceExampleCentroid, int *d_changedSinceLastIteration)
{
	int myExample = blockIdx.x * blockDim.x + threadIdx.x;
	if (myExample >= nExamples)
		return;

	float smallestDistanceToCentroid = FLT_MAX;
	int assignedCentroid;
	float distanceToCurrentCentroid;
	int jCentroid;
	for (jCentroid = 0; jCentroid < nCentroids; jCentroid++)
	{
		distanceToCurrentCentroid = d_distanceExampleCentroid[myExample * nCentroids + jCentroid];
		if (distanceToCurrentCentroid < smallestDistanceToCentroid) {
			smallestDistanceToCentroid = distanceToCurrentCentroid;
			assignedCentroid = jCentroid;
		}
	}

	if (d_centroidAssignedToExample[myExample] != assignedCentroid)
		*d_changedSinceLastIteration = 1;

	d_centroidAssignedToExample[myExample] = assignedCentroid;

	int i;
	atomicAdd(&d_numberOfExamplePerCentroid[assignedCentroid], 1);
	for (i = 0; i < nDim; i++) {
		atomicAdd(&d_runningSumOfExamplesPerCentroid[assignedCentroid * nDim + i], d_dataX[myExample * nDim + i]);
	}

}

__global__
void update_centroids(float *d_centroidPosition,
		float *d_runningSumOfExamplesPerCentroid,
		int *d_numberOfExamplePerCentroid, int nCentroids, int nDim) {

	int myExample = blockIdx.x * blockDim.x + threadIdx.x;

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

float* KmeansParallel::run(int nCentroids, int maxIter) {
	this->nCentroids = nCentroids;
	lastExecutionTime = 0;
	AllocateMemoryForCentroidVariables();

	//InitializeCentroids
	(*initializeCentroidsFunction)(dataX, centroidPosition, nCentroids, nDim,
			nExamples, verbose);

	CudaTimer totalTime;

	totalTime.start();
	AllocateMemoryAndCopyVariablesToGPU();

	int blockSize_examplesDim = BLOCKSIZEEXAMPLES;
	int gridSize_examplesDim = nExamples / blockSize_examplesDim + 1;

	int blockSize_centroidsDim = BLOCKSIZECENTROIDS;
	int gridSize_centroidsDim = nCentroids / blockSize_centroidsDim + 1;

    const dim3 blockSize_main (blockSize_examplesDim, blockSize_centroidsDim, 1);
    const dim3 gridSize_main (gridSize_examplesDim, gridSize_centroidsDim, 1);

	int blockSize_centroidsOnly = 32;
	int gridSize_centroidsOnly = nCentroids / blockSize_centroidsOnly + 1;

    const dim3 blockSize_centroids (blockSize_centroidsOnly,1, 1);
    const dim3 gridSizeCentroids(gridSize_centroidsOnly, 1, 1);

	int blockSize_examplesOnly = 32;
	int gridSize_examplesOnly = nExamples / blockSize_examplesOnly + 1;

    const dim3 blockSize_examples(blockSize_examplesOnly,1, 1);
    const dim3 gridSizeExamples(gridSize_examplesOnly, 1, 1);

    changedSinceLastIteration = 1;
    int nIteration = 0;

    float totalTimeInMainKernel = 0;
    float totalTimeInClearVectorsKernel = 0;
    float totalTimeInAggregateCentroidsKernel = 0;
    float totalTimeInUpdateCentroidsKernel = 0;

    CudaTimer cudaTimer;


    while (changedSinceLastIteration &&
    		(nIteration < maxIter || maxIter == -1)) {
    	if (verbose)
    		printf ("Starting iteration %d:\n", nIteration);

    	cudaTimer.start();
		clear_vectors<<<gridSizeCentroids, blockSize_centroids>>> (d_runningSumOfExamplesPerCentroid, d_numberOfExamplePerCentroid,nCentroids, nDim, d_changedSinceLastIteration);
		syncAndCheckErrors();
		totalTimeInClearVectorsKernel += cudaTimer.stop();

		cudaTimer.start();
		run_kmeans_parallel<<<gridSize_main, blockSize_main>>> (d_dataX, d_centroidPosition,
				d_centroidAssignedToExample, d_runningSumOfExamplesPerCentroid, d_numberOfExamplePerCentroid, nExamples, nCentroids, nDim, d_distanceExampleCentroid);
		syncAndCheckErrors();
		totalTimeInMainKernel += cudaTimer.stop();

		cudaTimer.start();
		aggregate_centroid_locations<<<gridSizeExamples, blockSize_examples>>> (d_runningSumOfExamplesPerCentroid, d_numberOfExamplePerCentroid, d_centroidAssignedToExample, d_dataX, nDim, nExamples, nCentroids, d_distanceExampleCentroid, d_changedSinceLastIteration);
		syncAndCheckErrors();
		totalTimeInAggregateCentroidsKernel += cudaTimer.stop();

		cudaTimer.start();
		update_centroids<<<gridSizeCentroids, blockSize_centroidsOnly>>> (d_centroidPosition,
				d_runningSumOfExamplesPerCentroid, d_numberOfExamplePerCentroid,nCentroids, nDim);
		syncAndCheckErrors();
		totalTimeInUpdateCentroidsKernel += cudaTimer.stop();

		CopyCompletionFlagFromGPU();

		if (nIteration == 0)
			changedSinceLastIteration = true;

		nIteration ++;
    }



    CopyResultsFromGPU();

    float totalTimeInKernels = totalTimeInClearVectorsKernel + totalTimeInMainKernel + totalTimeInAggregateCentroidsKernel + totalTimeInUpdateCentroidsKernel;
    
    float total = totalTime.stop();
    lastExecutionTime = total;

    if (verbose)
    {
		printf("Total time: %f ms\n", total);
		printf("Total kernel time: %f ms\n", totalTimeInKernels);
		printf("Time spent for each kernel: \n");
		printf("Clear: %f ms (%.2f%%)\n", totalTimeInClearVectorsKernel, totalTimeInClearVectorsKernel / totalTimeInKernels * 100);
		printf("Main: %f ms (%.2f%%)\n", totalTimeInMainKernel, totalTimeInMainKernel/ totalTimeInKernels *100);
		printf("Aggregate Centroids: %f ms (%.2f%%)\n", totalTimeInAggregateCentroidsKernel, totalTimeInAggregateCentroidsKernel/ totalTimeInKernels *100);
		printf("Update Centroids: %f ms (%.2f%%)\n", totalTimeInUpdateCentroidsKernel, totalTimeInUpdateCentroidsKernel/ totalTimeInKernels *100);
		printf("Centroids: \n");
		int i;
		for (i = 0; i < nCentroids; i++)
			printf("%.17g %.17g %.17g %.17g\n", centroidPosition[i * nDim + 0],
					centroidPosition[i * nDim + 1], centroidPosition[i * nDim + 2],
					centroidPosition[i * nDim + 3]);
		fflush(stdout);

    }

	FreeHostMemory();
	FreeGPUMemory();


	return centroidPosition;
}


KmeansParallel::KmeansParallel(float *data, int nExamples, int nDim,
		bool verbose) {
	this->dataX = data;
	this->nExamples = nExamples;
	this->nDim = nDim;
	this->verbose = verbose;
	this->initializeCentroidsFunction = &InitializeCentroids;
	centroidPosition = 0;
	lastExecutionTime = 0;
}

KmeansParallel::~KmeansParallel() {

}

float KmeansParallel::getLastRunningTime() {
	return lastExecutionTime;
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

void KmeansParallel::syncAndCheckErrors() {
	cudaDeviceSynchronize();
	checkCudaErrors (cudaGetLastError());}



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
					cudaMalloc(&d_distanceExampleCentroid, sizeof(float) * nExamples * nCentroids ));


	checkCudaErrors(
			cudaMemcpy(d_centroidPosition, centroidPosition, sizeof(float) * (nCentroids * nDim), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_dataX, dataX, sizeof(float) * (nExamples * nDim), cudaMemcpyHostToDevice));
}

void KmeansParallel::FreeHostMemory() {
	free(centroidPosition);
	free(centroidAssignedToExample);
	free(runningSumOfExamplesPerCentroid);
	free(numberOfExamplePerCentroid);
}

void KmeansParallel::FreeGPUMemory() {
	checkCudaErrors(cudaFree(d_dataX));
	checkCudaErrors(cudaFree(d_centroidPosition));
	checkCudaErrors(cudaFree(d_centroidAssignedToExample));
	checkCudaErrors(cudaFree(d_runningSumOfExamplesPerCentroid));
	checkCudaErrors(cudaFree(d_numberOfExamplePerCentroid));
	checkCudaErrors(cudaFree(d_changedSinceLastIteration));
	checkCudaErrors(cudaFree(d_distanceExampleCentroid));
}

void KmeansParallel::CopyCompletionFlagFromGPU(){
	checkCudaErrors(cudaMemcpy(&changedSinceLastIteration, d_changedSinceLastIteration, sizeof(int) , cudaMemcpyDeviceToHost));
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
		int nCentroids, int nDim, int nExamples, bool verbose) {
	//Initialize centroids with K random examples (Forgy's method)

    int *randomVector;
    int i,j;

    randomVector = (int*) malloc (sizeof(int) * nExamples);
    for (int i =0; i< nExamples;i++)
    	randomVector[i] = i;

    std::random_shuffle(randomVector, randomVector+nExamples);

    if(verbose)
    	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i = 0; i < nCentroids; i++) {
		selectedExample = randomVector[i];
		if(verbose)
			printf("%d ", selectedExample);
		for (j = 0; j < nDim; j++)
					centroidPosition[i * nDim + j] = dataX[selectedExample * nDim + j];
	}
	if(verbose)
		printf("\n");
	free(randomVector);
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
