/*
 * kmeans_tester.cpp
 *
 *  Created on: 16/07/2013
 *      Author: gustavo
 */

#include "kmeans_tester.h"

KmeansTester::KmeansTester(Kmeans *kmeans, int nRuns) {
	this->kmeans = kmeans;
	this->nRuns = nRuns;
}

float KmeansTester::RunAndGetAverageExecutionTime(int nCentroids) {
	float totalTime = 0;

	for(int i = 0; i < nRuns; i++)	{
		kmeans->run(nCentroids);
		totalTime += kmeans->getLastRunningTime();
	}

	return totalTime / nRuns;
}


