/*
 * kmenas_serial.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_SERIAL_H_
#define KMEANS_SERIAL_H_

class KmeansSerial {

private:
	bool verbose;
	double *dataX;

	double *centroidPosition;
	int *centroidAssignedToExample;
	double *runningSumOfExamplesPerCentroid;
	int *numberOfExamplePerCentroid;

	int nExamples;
	int nDim;
	int nCentroids;

	initFunction initializeCentroidsFunction;

	int FindClosestCentroidsAndCheckForChanges();
	void AllocateMemoryForCentroidVariables();
	void ClearIntArray(int* vector, int size);
	void ClearDoubleArray(double* vector, int size);
	static void InitializeCentroids(double *dataX, double *centroidPosition,int nCentroids, int nDim, int nExamples);
	double CalculateDistance(double *dataX, double *centroidPosition, int iExample,int jCentroid);
	int GetClosestCentroid(int iExample);
	void CompareTestResultsAgainstBaseline(double *centroidPosition);

public:
	KmeansSerial(double *data, int nExamples, int nDim, bool verbose = false);
	void setInitializeCentroidsFunction(initFunction fun);
	double* run(int nCentroids);
};


#endif /* KMEANS_SERIAL_H_ */
