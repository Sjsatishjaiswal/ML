
    
    
                Hyperparameter Tuning
                
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_wine
dataset = load_wine()

X = dataset.data
y = dataset.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
params = {'n_neighbors' : [1, 2, 3, 4, 5, 6, 7]}

grid = GridSearchCV(knn, params)
grid.fit(X, y)

best_knn = grid.best_estimator_

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

params = [{'criterion' : ['gini', 'entropy']},
          {'max_depth' : [3, 5, 9]}]

grid = GridSearchCV(dtf, params)
grid.fit(X, y)

best_tree = grid.best_estimator_

#############################################################

                    Ensemble Techniques
                    
Ensemble simply means a group. If we apply a group of ml
algorithms together on the same dataset/problem, the practice
is known as ensemble learning.

Ensemble technique can be applied in the following ways:
    1. Stacking (ML)
    2. Bagging (ML)
    3. Boosting (DL)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

from sklearn.svm import SVC
svm = SVC() 

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.ensemble import VotingClassifier
vote = VotingClassifier([('LR', log_reg),
                         ('KNN', knn),
                         ('DT', dtf),
                         ('SVM', svm),
                         ('NB', nb)])

vote.fit(X, y)
vote.score(X, y)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(nb, n_estimators = 8)

bag.fit(X, y)
bag.score(X, y)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 6)

rf.fit(X, y)
rf.score(X, y)


Stacking                      |                   Bagging
Different Algos               |   Different Implementation of same algo
Same Dataset                  |   Shuffled subset of the dataset

A special case of bagging technique where the base_estimator is a 
Decision Tree is known as Random Forest Algorithm.






















































