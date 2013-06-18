#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "iris_data.h"
//#include "load_cifar10_data.h"
#include "kmeans.h"
#include "kmeans_serial.h"
#include "kmeans_parallel.h"

void compareVectors(float * centroidPosition1, float * baseline, int nCentroids, int nDim);
void compareIntVectors(int * centroidAssignedtoExample1, int * baseline, int nExamples);
void PrintVector(int *v, int size);
void PrintVector(float *v, int size);

void InitializeCentroidsTest(float *dataX, float *centroidPosition,
		int nCentroids, int nDim, int) {
	//Initialize centroids with K random examples (Forgy's method)
	int i;
	int selectedExamples[] = { 16, 85, 79 };
	printf("Centroids initialized with examples: ");
	int selectedExample;
	for (i = 0; i < nCentroids; i++) {
		selectedExample = selectedExamples[i];
		printf("%d ", selectedExample);
		centroidPosition[i * nDim + 0] = dataX[selectedExample * nDim + 0];
		centroidPosition[i * nDim + 1] = dataX[selectedExample * nDim + 1];
		centroidPosition[i * nDim + 2] = dataX[selectedExample * nDim + 2];
		centroidPosition[i * nDim + 3] = dataX[selectedExample * nDim + 3];
	}
	printf("\n");

}

void CompareTestResultsAgainstBaseline(float *centroidPosition, int nDim) {
	int nCentroids = 3;
	float baseline[] = { 5.0059995651245117, 3.4180002212524414,1.4639999866485596, 0.24399997293949127,
			6.853844165802002, 3.0769233703613281, 5.7153849601745605, 2.0538463592529297,
			5.8836064338684082, 2.7409837245941162, 4.3885250091552734, 1.4344264268875122 };
	int i;
	float maxError = 1e-5;
	float error = 0;
	for (i = 0; i < nCentroids * nDim; i++) {
		error = fabs(centroidPosition[i] - baseline[i]);
		assert(error < maxError);
	}

	printf("OK!! Error agains baseline below threshold: %lf\n", error);
}

void testExecutionOnIrisDataset()
{
	float *centroidPositions;
	IrisDataset d;

	KmeansParallel kmeans (d.X, d.nExamples, d.nDim, true);
	kmeans.setInitializeCentroidsFunction(InitializeCentroidsTest);
	centroidPositions = kmeans.run(3);

	CompareTestResultsAgainstBaseline(centroidPositions, d.nDim);
}

void testOneIteration()
{
	float *centroidPositions_parallel, *centroidPositions_serial;
	IrisDataset d;

	KmeansParallel kmeans_parallel (d.X, d.nExamples, d.nDim, true);
	kmeans_parallel.setInitializeCentroidsFunction(InitializeCentroidsTest);
	centroidPositions_parallel = kmeans_parallel.run(3, 1);

	KmeansSerial kmeans_serial (d.X, d.nExamples, d.nDim, true);
	kmeans_serial.setInitializeCentroidsFunction(InitializeCentroidsTest);
	centroidPositions_serial = kmeans_serial.run(3, 1);

	compareVectors(centroidPositions_parallel, centroidPositions_serial, 3, d.nDim);
}

int main() {
	testOneIteration();

	testExecutionOnIrisDataset();
}

void compareVectors(float * centroidPosition1, float * baseline, int nCentroids, int nDim)
{
	int i;
	float maxError = 1e-3;
	float error = 0;
	for (i = 0; i < nCentroids * nDim; i++)	{
		error = fabs(centroidPosition1[i] - baseline[i]);
		if (error >= maxError)
			printf("NOK\nError in position i = %d. Values: %f and %f\n", i, centroidPosition1[i], baseline[i]);
		assert(error < maxError);
	}
			
}

void PrintVector(int *v, int size)
{
	for (int i =0; i<size;i++)
		printf("%d ",v[i]);
}


void PrintVector(float *v, int size)
{
	for (int i =0; i<size;i++)
		printf("%f ",v[i]);
}

void compareIntVectors(int * centroidAssignedtoExample1, int * baseline, int nExamples)
{
	int i;

	for (i = 0; i < nExamples; i++)	{
		assert(centroidAssignedtoExample1[i] == baseline[i]);
	}

}

