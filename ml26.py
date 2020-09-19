
                Decision Tree Algorithm
                
Decision : Prediction involves decision making
Tree : Non-linear data structure
Algorithm : Aap jaante he hai

The DT Algorithm develops a tree like structure in accordance
with the data fed to it. This tree like structure can then
be traversed for prediction.

Q.1. (k, tk) = ?
Q.2. depth = ?

Ans: The DT algorithm uses some measure of impurity to create
an optimal split. The algorithm then attempts to minimize a 
cost function for multiple pairs of (k, tk) resulting in a
pair that minimizes the aforementioned cost function.
The algorithm can then use this information to create and
save the tree for classification.

The DT algo popularly uses these measure of impurities:
    1. Gini Index
    2. Entropy
    
The DT algorithm is recursive in nature and often overfits the
training data. This is because if the depth is not explicitly
specified by the programmer, the DT algo goes on creating
a tree unless and until it can correctly catch all the training
observations.

Complexity of DT in sklean is O (log (m * n))

We split the original dataset into two parts:
    1. Training Set
    2. Testing Set
    
If the algorithm performs surprisingly well on training set
and poorly on test set then we call this process as Overfitting.

If the algorithm performs surprisingly poor on training set
and poorly on test set then we call this process as Underfitting.

###############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(criterion = 'entropy',
                             max_depth = 3)
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)

##############################################################


from sklearn.datasets import load_wine
dataset = load_wine()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)

#############################################################

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

X = dataset.data
y = dataset.target
y = y.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 15)

log_reg.fit(X_train, y_train)
# knn.fit(X_train, y_train)
dtf.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)

# knn.score(X_train, y_train)
# knn.score(X_test, y_test)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)




















































































