/*
 * kmeans.h
 *
 *  Created on: 09/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_H_
#define KMEANS_H_

typedef void (*initFunction)(double *, double *, int, int, int);
typedef int (*loadDataFunction)(double **X, int **y);


void InitializeCentroids (double *dataX, double *centroidPosition, int nCentroids);
void InitializeCentroidsTest (double *dataX, double *centroidPosition, int nCentroids);
double CalculateDistance(double *dataX, double *centroidPosition, int iExample, int jCentroid);
int GetClosestCentroid (double *dataX, double *centroidPosition, int iExample, int nCentroids);
void ClearIntArray(int *vector, int size);
void ClearDoubleArray(double *vector, int size);
void CompareTestResultsAgainstBaseline (double *centroidPosition);
int load_iris_data(double **X, int **y);

double *run_kmeans(initFunction InitializeCentroidsFunction,
				   loadDataFunction LoadDataFunction,
				   int nCentroids);


#endif /* KMEANS_H_ */