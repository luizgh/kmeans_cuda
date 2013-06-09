kmeans_cuda
===========

Kmeans implementation in C / Cuda


This repository contains the implementation of standard K-means:

1) Serial version
2) Parallel version using Cuda

It also contains code to load the following libraries:

1) The "Iris" dataset: http://archive.ics.uci.edu/ml/datasets/Iris
2) The CIFAR10 dataset: http://www.cs.toronto.edu/~kriz/cifar.html

In order to compile this code you need the CUDA toolkit and developer drivers (https://developer.nvidia.com/cuda-downloads)


How to use:

./run_kmeans <iris|cifar> <nCentroids>

Examples:

./run_kmeans cifar 100
