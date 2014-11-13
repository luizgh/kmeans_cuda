kmeans_cuda
===========

Kmeans implementation in C++ / CUDA


This repository contains the implementation of standard K-means:

* Serial version
* Parallel version using CUDA

It also contains code to load the following datasets:

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
