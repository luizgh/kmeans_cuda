NVCC=nvcc

SHAREDMEMORY=-DUSESHAREDMEMORY  #Note: TEST only: use it only for tests on the wine database, with centroids multiple of 30
#FIXSEED=-DFIXSEED

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64  -g $(FIXSEED)  -O3 $(SHAREDMEMORY) -lrt
GCC_OPTS=-Wall -Wextra -m64 -g $(FIXSEED) -O3 -lrt

all: bin/test_kmeans_serial bin/test_kmeans_parallel bin/run_kmeans bin/run_performance_test

bin/test_kmeans_parallel: obj/test_kmeans_parallel.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/kmeans_serial.o obj/cudaTimer.o obj/wine_data.o
	$(NVCC) -o bin/test_kmeans_parallel obj/test_kmeans_parallel.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/kmeans_serial.o obj/wine_data.o obj/cudaTimer.o $(NVCC_OPTS)

bin/test_kmeans_serial: obj/test_kmeans_serial.o obj/kmeans_serial.o obj/cifar10_data.o obj/iris_data.o  obj/wine_data.o
	g++ -o bin/test_kmeans_serial obj/test_kmeans_serial.o obj/kmeans_serial.o obj/cifar10_data.o obj/wine_data.o obj/iris_data.o $(GCC_OPTS)
	
bin/run_kmeans: obj/run_kmeans.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/cudaTimer.o obj/wine_data.o obj/kmeans_serial.o
	$(NVCC) -o bin/run_kmeans obj/run_kmeans.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/cudaTimer.o obj/wine_data.o obj/kmeans_serial.o $(NVCC_OPTS)

bin/run_performance_test: obj/run_performance_test.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/cudaTimer.o obj/wine_data.o obj/kmeans_serial.o obj/kmeans_tester.o
	$(NVCC) -o bin/run_performance_test obj/run_performance_test.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o obj/cudaTimer.o obj/wine_data.o obj/kmeans_serial.o obj/kmeans_tester.o $(NVCC_OPTS)

obj/run_kmeans.o: src/run_kmeans.cpp
	$(NVCC) -c -o obj/run_kmeans.o src/run_kmeans.cpp $(NVCC_OPTS)

obj/run_performance_test.o: src/run_performance_test.cpp
	$(NVCC) -c -o obj/run_performance_test.o src/run_performance_test.cpp $(NVCC_OPTS)

obj/kmeans_tester.o: src/kmeans_tester.cpp
	$(NVCC) -c -o obj/kmeans_tester.o src/kmeans_tester.cpp $(NVCC_OPTS)

obj/test_kmeans_parallel.o: src/test_kmeans_parallel.cpp 
	$(NVCC) -c -o obj/test_kmeans_parallel.o src/test_kmeans_parallel.cpp $(NVCC_OPTS)

obj/test_kmeans_serial.o: src/test_kmeans_serial.cpp 
	g++ -c -o obj/test_kmeans_serial.o src/test_kmeans_serial.cpp $(GCC_OPTS)

obj/kmeans_parallel.o: src/kmeans_parallel.cu src/kmeans_parallel.h
	$(NVCC) -c -o obj/kmeans_parallel.o src/kmeans_parallel.cu $(NVCC_OPTS)

obj/kmeans_serial.o: src/kmeans_serial.cpp src/kmeans_serial.h
	g++ -c -o obj/kmeans_serial.o src/kmeans_serial.cpp $(GCC_OPTS)

obj/cifar10_data.o: src/cifar10_data.cpp src/cifar10_data.h
	g++ -c -o obj/cifar10_data.o src/cifar10_data.cpp $(GCC_OPTS)

obj/iris_data.o: src/iris_data.cpp src/iris_data.h
	g++ -c -o obj/iris_data.o src/iris_data.cpp $(GCC_OPTS)

obj/wine_data.o: src/wine_data.cpp src/wine_data.h
	g++ -c -o obj/wine_data.o src/wine_data.cpp $(GCC_OPTS)


obj/cudaTimer.o: src/cudaTimer.cpp src/cudaTimer.h
	$(NVCC) -c -o obj/cudaTimer.o src/cudaTimer.cpp $(NVCC_OPTS)
clean:
	rm -f obj/*.o bin/run_kmeans_parallel bin/run_kmeans_serial bin/test_kmeans
