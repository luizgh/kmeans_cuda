/*
 * load_cifar10_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef CIFAR10_DATA_H_
#define CIFAR10_DATA_H_

#include "dataset.h"

class Cifar10Dataset_1batch : public Dataset {

private:
	int load_cifar_data_from_batches();
	static const int nExamples_perbatch = 10000;
	void processBatch(int batchNumber);
public:
	Cifar10Dataset_1batch();
	~Cifar10Dataset_1batch();
	float *X;
	int *y;
	static const int nExamples = 10000;
	static const int nDim = 3072;

	float* getX() { return X; }
	int* getY() { return y; }
	int getNExamples() { return nExamples; }
	int getnDim() { return nDim; }


};

#endif /* CIFAR10_DATA_H_ */
