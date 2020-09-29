
    
                            Fine Tuning
                            
Fine: Miniscule and carefully done
Tuning: Process of iteratively improving an existing model

1. You should never use the entire data for training an ML algorithm.

"A good ml model must perform better in the real world, not just \
    on the data it has been trained on. We take out a small part \
        of the dataset and keep it safe. When the model is trained \
            on the remaining data, this safe set (test set) can be \
                used as a real world scenario to check how good \
                    an algo is performing on the data it has never \
                        seen before."
                        
2. Apply K-fold Cross Validation

In this technique we split the training set into 'K' equal parts
and train a particular algorithm on '(k-1)' parts followed by testing
the same algorithm on the 'Kth part.' This process is repeated 'K' 
times.


#################### Ab pehle practical ###########################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7)

### Ab Test Set ko Bhul Jao ###

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 3)
dtf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(dtf, X_train, y_train, cv = 3)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

cross_val_score(log_reg, X_train, y_train)

#### Ab Khelo apni ML knowledge k saath aur jo jo sikha hai
### sab karol

### Best model Logistic Wala Tha

log_reg.score(X_test, y_test)

The test set accuracy turns out to be closely related to
training set and validation set accuracies so it is a good
model.

But suppose this was the case

[0.94186047, 0.87058824, 0.97647059, 0.96470588, 0.95294118]

Test Set : 0.23 ----> Overfitting

English mein, the model is performing very well on the training
set but performs poorly on the real world dataset that it has
not seen before.












































































