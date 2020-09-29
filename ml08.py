
    
                Unsupervised Learning
                
Unsupervised learning is a subset of data mining
and machine learning that processes huge chunks of 
data to gather interesting insights.

A common unsupervised learning technique is clustering.

Clustering refers to the process of creating 
homogenous clusters in the dataset.

There are various clustering algorithms but the most
used one is known as K-Means Clustering.

                K - Means Clustering
                
K : Arbitrary number
Means : Averages
Clustering : Making clusters

Algorithm

1. Initialize 'K' centroids randomly in the dataset.
2. For all the observations 'm' find out the closest
    centroid (euclidean) and form a cluster.
3. For the newly formed cluster, compute the centre of
    gravity (COGs).
4. Initialize the COGs as new centroids and repeat from
    step 2.
5. The algorithm will terminate when points stop
    rearrangement.

##########################################################
    
Q. How does the K-Means clustering algorithm deduces the
optimal number of K?

Ans: WCV (Within Cluster Variation)

#########################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.6)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 11):
    km = KMeans(n_clusters = i)
    km.fit(x)
    wcv.append(km.inertia_)
    
plt.plot(range(1, 11), wcv)
plt.plot()

km = KMeans(n_clusters = 5)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], c = "r", label = 'Target Customer')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], c = "g", label = 'Target Customer')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], c = "b", label = 'Smart Buyers')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], c = "y", label = 'Smart Buyers')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], c = "magenta", label = 'Target Customer')
plt.xlabel('Salary')
plt.ylabel('Expense')
plt.title('K-Means Practical | Supermarket Analysis')
plt.legend()
plt.show()




















































































