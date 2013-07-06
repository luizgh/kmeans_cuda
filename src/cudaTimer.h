/*
 * cudaTimer.h
 *
 *  Created on: 05/07/2013
 *      Author: gustavo
 */

#ifndef CUDATIMER_H_
#define CUDATIMER_H_

#include <cuda.h>
#include <cuda_runtime.h>

class CudaTimer {
private:
	cudaEvent_t startEvent, stopEvent;
	float time;

public:
	CudaTimer();
	virtual ~CudaTimer();
	void start();
	float stop();
	float getElapsedTime();
};

#endif /* CUDATIMER_H_ */

