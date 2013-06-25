/*
 * kmenas_serial.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef KMEANS_PARALLEL_H_
#define KMEANS_PARALLEL_H_

class KmeansParallel {

public:
	bool verbose;
	float *dataX;
	float *centroidPosition;
	int *centroidAssignedToExample;
	float *runningSumOfExamplesPerCentroid;
	int *numberOfExamplePerCentroid;
	int changedSinceLastIteration;

	float *d_dataX;
	float *d_centroidPosition;
	int *d_centroidAssignedToExample;
	float *d_runningSumOfExamplesPerCentroid;
	int *d_numberOfExamplePerCentroid;
	int* d_changedSinceLastIteration;


	int nExamples;
	int nDim;
	int nCentroids;

	initFunction initializeCentroidsFunction;

	int FindClosestCentroidsAndCheckForChanges();
	void AllocateMemoryForCentroidVariables();
	void AllocateMemoryAndCopyVariablesToGPU();
	void CopyResultsFromGPU();
	void ClearIntArray(int* vector, int size);
	void ClearfloatArray(float* vector, int size);
	static void InitializeCentroids(float *dataX, float *centroidPosition,int nCentroids, int nDim, int nExamples);
	float CalculateDistance(float *dataX, float *centroidPosition, int iExample,int jCentroid);
	int GetClosestCentroid(int iExample);
	void CompareTestResultsAgainstBaseline(float *centroidPosition);
	void CopyCompletionFlagFromGPU();
	void FreeHostMemory();
	void FreeGPUMemory();

public:
	KmeansParallel(float *data, int nExamples, int nDim, bool verbose = false);
	~KmeansParallel();
	void setInitializeCentroidsFunction(initFunction fun);
	float* run(int nCentroids, int maxIter = -1);
};




#endif /* KMEANS_PARALLEL_H_ */
