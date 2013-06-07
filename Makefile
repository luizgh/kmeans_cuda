NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64


bin/kmeans_parallel: obj/kmeans_parallel.o obj/load_cifar10_data.o obj/load_iris_data.o
	$(NVCC) -o obj/kmeans_parallel obj/kmeans_parallel.o $(NVCC_OPTS)

bin/kmeans_serial: obj/kmeans_serial.o obj/load_iris_data.o obj/load_cifar10_data.o
	gcc -o bin/kmeans_serial obj/kmeans_serial.o obj/load_iris_data.o obj/load_cifar10_data.o -lm $(GCC_OPTS)

obj/kmeans_parallel.o: kmeans_parallel.cu kmeans_parallel.h
	$(NVCC) -c -o obj/kmeans_parallel.o kmeans_parallel.cu $(NVCC_OPTS)

obj/kmeans_serial.o: kmeans_serial.c kmeans_serial.h
	gcc -c -o obj/kmeans_serial.o kmeans_serial.c $(GCC_OPTS)

obj/load_cifar10_data.o: load_cifar10_data.c load_cifar10_data.h
	gcc -c -o obj/load_cifar10_data.o load_cifar10_data.c $(GCC_OPTS)

obj/load_iris_data.o: load_iris_data.c load_iris_data.h
	gcc -c -o obj/load_iris_data.o load_iris_data.c $(GCC_OPTS)

clean:
	rm -f obj/*.o bin/kmeans_parallel bin/kmeans_serial
