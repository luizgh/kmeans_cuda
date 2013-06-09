/*
 * load_iris_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef LOAD_IRIS_DATA_H_
#define LOAD_IRIS_DATA_H_

class IrisDataset {

private:
	int load_iris_data_from_file(const char *filename);
public:
	IrisDataset();
	float *X;
	int *y;
	static const int nExamples = 150;
	static const int nDim = 4;

};





#endif /* LOAD_IRIS_DATA_H_ */
