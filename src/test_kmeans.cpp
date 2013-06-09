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

void InitializeCentroidsTest(double *dataX, double *centroidPosition,
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

void CompareTestResultsAgainstBaseline(double *centroidPosition, int nDim) {
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

int main() {
	double *centroidPositions;
	IrisDataset d;

	KmeansParallel kmeans (d.X, d.nExamples, d.nDim, true);
	kmeans.setInitializeCentroidsFunction(InitializeCentroidsTest);
	centroidPositions = kmeans.run(3);

	CompareTestResultsAgainstBaseline(centroidPositions, d.nDim);


}



