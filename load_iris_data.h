/*
 * load_iris_data.h
 *
 *  Created on: 04/06/2013
 *      Author: gustavo
 */

#ifndef LOAD_IRIS_DATA_H_
#define LOAD_IRIS_DATA_H_

#define NEXAMPLES 150
#define NDIM 4

static char iris_classes[3][20] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
int load_iris_data_from_file(char *filename, double **X, int **y);



#endif /* LOAD_IRIS_DATA_H_ */
