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
	for (i = 0; i < nCentroids * nDim; i++)
		error += fabs(centroidPosition[i] - baseline[i]);

	assert(error < maxError);
	printf("OK!! Error agains baseline below threshold: %lf\n", error);
}

int main() {

	float *centroidPositions;
	IrisDataset d;

	KmeansSerial kmeans (d.X, d.nExamples, d.nDim, true);
	kmeans.setInitializeCentroidsFunction(InitializeCentroidsTest);
	centroidPositions = kmeans.run(3);

	CompareTestResultsAgainstBaseline(centroidPositions, d.nDim);


}



