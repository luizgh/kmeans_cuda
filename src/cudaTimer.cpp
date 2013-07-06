/*
 * cudaTimer.cpp
 *
 *  Created on: 05/07/2013
 *      Author: gustavo
 */

#include "cudaTimer.h"

CudaTimer::CudaTimer() {
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
}

CudaTimer::~CudaTimer() {
	cudaEventDestroy( startEvent );
	cudaEventDestroy( stopEvent );
}



void CudaTimer::start() {
	time = 0;
	cudaEventRecord( startEvent, 0 );
}
float CudaTimer::stop() {
	cudaEventRecord( stopEvent, 0 );
	cudaEventSynchronize( stopEvent );

	cudaEventElapsedTime( &time, startEvent, stopEvent );
	return time;
}

float CudaTimer::getElapsedTime() {
	return time;
}
