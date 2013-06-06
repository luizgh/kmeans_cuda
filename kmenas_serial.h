/*
 * kmenas_serial.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef KMENAS_SERIAL_H_
#define KMENAS_SERIAL_H_

void InitializeCentroids (double *dataX, double *centroidPosition, int nCentroids);
void InitializeCentroidsTest (double *dataX, double *centroidPosition, int nCentroids);
double CalculateDistance(double *dataX, double *centroidPosition, int iExample, int jCentroid);
int GetClosestCentroid (double *dataX, double *centroidPosition, int *centroidAssignedToExample, int iExample, int nCentroids);
void ClearIntArray(int *vector, int size);
void ClearDoubleArray(double *vector, int size);
double* run_kmeans();
void CompareTestResultsAgainstBaseline (double *centroidPosition);
void load_iris_data(double **X, int **y);
#endif /* KMENAS_SERIAL_H_ */
