/*
 * kmeans.h
 *
 *  Created on: 09/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_H_
#define KMEANS_H_

typedef void (*initFunction)(float *, float *, int, int, int);
typedef int (*loadDataFunction)(float **X, int **y);


void InitializeCentroids (float *dataX, float *centroidPosition, int nCentroids);
void InitializeCentroidsTest (float *dataX, float *centroidPosition, int nCentroids);
float CalculateDistance(float *dataX, float *centroidPosition, int iExample, int jCentroid);
int GetClosestCentroid (float *dataX, float *centroidPosition, int iExample, int nCentroids);
void ClearIntArray(int *vector, int size);
void ClearfloatArray(float *vector, int size);
void CompareTestResultsAgainstBaseline (float *centroidPosition);
int load_iris_data(float **X, int **y);

float *run_kmeans(initFunction InitializeCentroidsFunction,
				   loadDataFunction LoadDataFunction,
				   int nCentroids);


#endif /* KMEANS_H_ */
