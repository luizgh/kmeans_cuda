/*
 * run_performance_test.cpp
 *
 *  Created on: 16/07/2013
 *      Author: gustavo
 */

#include <cstdio>
#include "kmeans_tester.h"
#include "kmeans_serial.h"
#include "kmeans_parallel.h"
#include "wine_data.h"

const int nRuns = 20;

int main()
{

	Dataset *dataset = new WineDataset;

	Kmeans *kmeansSerial = new KmeansSerial(dataset-> getX(), dataset->getNExamples(), dataset->getnDim(), false);
	Kmeans *kmeansParallel = new KmeansParallel(dataset-> getX(), dataset->getNExamples(), dataset->getnDim(), false);

	KmeansTester testSerial(kmeansSerial, nRuns);
	KmeansTester testParallel(kmeansParallel, nRuns);

	int testCentroids[] = { 30, 60, 90, 120,150};
	int nTests = sizeof(testCentroids) / sizeof(int);

	printf("NCentroids\tSerial\tParallel\n");
	for(int i = 0; i < nTests; i++)
	{
		float serialTime = testSerial.RunAndGetAverageExecutionTime(testCentroids[i]);
		float parallelTime = testParallel.RunAndGetAverageExecutionTime(testCentroids[i]);
		printf("%d\t%f\t%f\n",testCentroids[i], serialTime, parallelTime);
	}

}

