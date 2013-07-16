/*
 * kmeans_tester.h
 *
 *  Created on: 16/07/2013
 *      Author: gustavo
 */

#ifndef KMEANS_TESTER_H_
#define KMEANS_TESTER_H_

#include "kmeans.h"
#include "dataset.h"

class KmeansTester {
private:
	Kmeans *kmeans;
	int nRuns;
public:
	KmeansTester(Kmeans *kmeans, int nRuns);

	float RunAndGetAverageExecutionTime(int nCentroids);
};


#endif /* KMEANS_TESTER_H_ */
