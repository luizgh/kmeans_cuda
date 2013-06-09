NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 #-O3
GCC_OPTS=-Wall -Wextra -m64 -g #-O3

all: bin/run_kmeans bin/test_kmeans

bin/test_kmeans: obj/test_kmeans.o obj/kmeans_serial.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o
	$(NVCC) -o bin/test_kmeans obj/test_kmeans.o obj/kmeans_serial.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o $(NVCC_OPTS)
	
bin/run_kmeans: obj/run_kmeans.o obj/kmeans_serial.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o
	$(NVCC) -o bin/run_kmeans obj/run_kmeans.o obj/kmeans_serial.o obj/kmeans_parallel.o obj/cifar10_data.o obj/iris_data.o $(NVCC_OPTS)

obj/run_kmeans.o: src/run_kmeans.cpp
	g++ -c -o obj/run_kmeans.o src/run_kmeans.cpp $(GCC_OPTS)

obj/test_kmeans.o: src/test_kmeans.cpp 
	g++ -c -o obj/test_kmeans.o src/test_kmeans.cpp $(GCC_OPTS)

obj/kmeans_parallel.o: src/kmeans_parallel.cu src/kmeans_parallel.h
	$(NVCC) -c -o obj/kmeans_parallel.o src/kmeans_parallel.cu $(NVCC_OPTS)

obj/kmeans_serial.o: src/kmeans_serial.cpp src/kmeans_serial.h
	g++ -c -o obj/kmeans_serial.o src/kmeans_serial.cpp $(GCC_OPTS)

obj/cifar10_data.o: src/cifar10_data.cpp src/cifar10_data.h
	g++ -c -o obj/cifar10_data.o src/cifar10_data.cpp $(GCC_OPTS)

obj/iris_data.o: src/iris_data.cpp src/iris_data.h
	g++ -c -o obj/iris_data.o src/iris_data.cpp $(GCC_OPTS)

clean:
	rm -f obj/*.o bin/kmeans_parallel bin/kmeans_serial
