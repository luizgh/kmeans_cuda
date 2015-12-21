K-means implementation in C++ / CUDA
===========

Kmeans implementation in C++ / CUDA. I implemented this project for an assignment to a "Parallel programming" course during my master's. 

This repository contains a serial implementation of k-means (in C++) and a parallel implementation for running on the GPU (CUDA). The serial version is about 2x faster than the Scipy's implementation. The parallel implementation is 18x faster than Scipy's implementation, but the algorithm uses O(n^2) memory. Further improvements would be required to keep the gains in speed while keeping the memory usage linear. The report [can be found here](https://github.com/luizgh/kmeans_cuda/blob/master/relatorio.pdf) (written in Portuguese)

This repository also contains code to load the following datasets:

* The "Iris" dataset: http://archive.ics.uci.edu/ml/datasets/Iris
* The CIFAR10 dataset: http://www.cs.toronto.edu/~kriz/cifar.html

In order to compile this code you need the CUDA toolkit and developer drivers (https://developer.nvidia.com/cuda-downloads)

How to use:
```
  ./run_kmeans <iris|cifar> <nCentroids>
```
Examples:
```
  ./run_kmeans cifar 100
```
