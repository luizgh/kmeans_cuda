/*
 * Dataset.h
 *
 *  Created on: 09/06/2013
 *      Author: gustavo
 */

#ifndef DATASET_H_
#define DATASET_H_

class Dataset {
public:
	double *X;
	int *y;
	virtual Dataset();

	virtual ~Dataset();
};

#endif /* DATASET_H_ */
