/*
 * kmenas_serial.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_SERIAL_H_
#define KMEANS_SERIAL_H_

class KmeansSerial {

	//TODO: switch back to private after tests
public:
	bool verbose;
	float *dataX;

	float *centroidPosition;
	int *centroidAssignedToExample;
	float *runningSumOfExamplesPerCentroid;
	int *numberOfExamplePerCentroid;

	int nExamples;
	int nDim;
	int nCentroids;

	initFunction initializeCentroidsFunction;

	int FindClosestCentroidsAndCheckForChanges();
	void AllocateMemoryForCentroidVariables();
	void ClearIntArray(int* vector, int size);
	void ClearfloatArray(float* vector, int size);
	static void InitializeCentroids(float *dataX, float *centroidPosition,int nCentroids, int nDim, int nExamples);
	float CalculateDistance(float *dataX, float *centroidPosition, int iExample,int jCentroid);
	int GetClosestCentroid(int iExample);
	void CompareTestResultsAgainstBaseline(float *centroidPosition);

public:
	KmeansSerial(float *data, int nExamples, int nDim, bool verbose = false);
	void setInitializeCentroidsFunction(initFunction fun);
	float* run(int nCentroids, int maxIter = -1);
};


#endif /* KMEANS_SERIAL_H_ */
