
        KNN (K - Nearest Neighbors)
        
K : An arbitrary number
Nearest : Distance Adjective
Neighbors : Observations closer to the point in
            consideration
            
KNN is a deterministic classification based 
algorithm that works for small and simple datasets.
The preferred distance metric is euclidean.
The value of 'k' must be carefully chosen and
experimented with.
The KNN algorithm can easily work with n-ary
classification problems.
KNN is an instance based technique with an
extremely large space and time complexity.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_wine
dataset = load_wine()

X = dataset.data
y = dataset.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X, y)

knn.score(X, y)

########## Handwritten Digits Image Classification ##########

Images in CS are nothing but a 2-D collection of pixels.











































































