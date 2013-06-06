/*
 * load_cifar10_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef LOAD_CIFAR10_DATA_H_
#define LOAD_CIFAR10_DATA_H_

#define NEXAMPLES_PER_BATCH 10000
#define NEXAMPLES 50000
#define NDIM 3072

int load_cifar_data_from_batches(double **X, int **y);

#endif /* LOAD_CIFAR10_DATA_H_ */
