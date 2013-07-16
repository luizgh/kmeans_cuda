/*
 * load_iris_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef DATASET_H_
#define DATASET_H_

class Dataset {

public:

	virtual float* getX() = 0;
	virtual int* getY() = 0;
	virtual int getNExamples() = 0;
	virtual int getnDim() = 0;

};





#endif
