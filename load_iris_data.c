#include <stdio.h>
#include <stdlib.h>
#include "load_iris_data.h"
/*load iris data into a matrix X, and vector y. Sample data format:
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
*/

int load_iris_data_from_file(char *filename, double **X, int **y)
//Expected:
//filename: string pointing to the flat file with iris data
//X: vector of size: sizeof(double) * numExamples * 4
//Y: vector of size: sizeof(int) * numExamples
{
	*X = (double*) malloc (sizeof(double) * (NEXAMPLES * NDIM));
	*y = (int*) malloc (sizeof(int) * (NEXAMPLES));

	FILE *file = fopen(filename,"r");
	if (!file) return 0;

	char line[256];
	double tempx[4];
	char class[100];
	int i = 0;
	int j;

	while (fgets(line, sizeof(line), file))
	{
		//Get line content
		sscanf(line, "%lf,%lf,%lf,%lf,%s", &tempx[0], &tempx[1], &tempx[2], &tempx[3], class);

		//Assign X
		(*X)[i*4 + 0] = tempx[0];
		(*X)[i*4 + 1] = tempx[1];
		(*X)[i*4 + 2] = tempx[2];
		(*X)[i*4 + 3] = tempx[3];

		//Assign y
		for (j = 0; j< 3; j++)
			if (strcmp(iris_classes[j], class) == 0)
				(*y)[i] = j;
		i++;
	}
	fclose(file);
	return 1;
}
