
import csv
import numpy
from scipy.cluster.vq import vq, kmeans, whiten

data = []

for file in ["winequality-red.csv", "winequality-white.csv"]:
  csvfile = open(file)
  reader = csv.reader(csvfile, delimiter = ';', quotechar='"')
  reader.next() 
  for row in reader:
    floatrow = [float(i) for i in row]
    data.append(floatrow)

dataset = numpy.asarray(data)

%timeit -n 100 kmeans(dataset, 900, iter=1, thresh=0)
%timeit -n 100 kmeans(dataset, 1200, iter=1, thresh=0)
%timeit -n 100 kmeans(dataset, 1500, iter=1, thresh=0)
%timeit -n 100 kmeans(dataset, 1800, iter=1, thresh=0)
%timeit -n 100 kmeans(dataset, 2100, iter=1, thresh=0)


