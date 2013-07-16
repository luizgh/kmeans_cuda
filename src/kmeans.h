/*
 * kmeans.h
 *
 *  Created on: 09/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_H_
#define KMEANS_H_

typedef void (*initFunction)(float *, float *, int, int, int);

class Kmeans
{
public:
	virtual void setInitializeCentroidsFunction(initFunction fun) = 0;
	virtual float* run(int nCentroids, int maxIter = -1) = 0;
};



#endif /* KMEANS_H_ */
