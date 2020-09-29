
    
                Dimensionality Reduction
                
In supervised learning, we mostly deal with tabular datasets which
are represented as 2d collection of rows and columns.

rows = m
columns = n

In a dataset, we can increase both rows and columns but only one
of them is problematic. If we increase rows, it means we are
getting more data which is a good thing. But if we increase
columns or features then we are increasing the complexity of
our model. 

This phenomenon is known as curse of dimensionality.

To solve the problem of curse of dimensionality we will use a
powerful dimensionality reduction algorithm known as PCA.

            PCA (Principal Component Analysis)
            
Principal : Main or Most Important
Component : Features or Columns
Analysis : Aap jaante he hai

PCA uses a matrix factorization technique known as singular
value decomposition (SVD) to divide the feature matrix into
the dot product of three matrices. One of which gives the 
principal features. The advantage of PCA is that it attempts to
preserve maximum variance in the data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)

X_3 = pca.fit_transform(X)
X_3

pca1 = PCA(n_components = 2)

X_2 = pca1.fit_transform(X)
X_2

pca2 = PCA(n_components = 1)

X_1 = pca2.fit_transform(X)
X_1

log_reg.fit(X_3, y)
log_reg.score(X_3, y)

log_reg.fit(X_2, y)
log_reg.score(X_2, y)

log_reg.fit(X_1, y)
log_reg.score(X_1, y)




















































