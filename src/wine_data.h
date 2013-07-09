/*
 * load_iris_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef LOAD_WINE_DATA_H_
#define LOAD_WINE_DATA_H_

class WineDataset {

private:
	int load_wine_data_from_file(const char *filename_red, const char *filename_white);
	int index;
	int read_file(const char *filename);

public:
	WineDataset();
	~WineDataset();
	float *X;
	int *y;
	static const int nExamples = 6497;
	static const int nDim = 11;

};





#endif /* LOAD_WINE_DATA_H_ */
