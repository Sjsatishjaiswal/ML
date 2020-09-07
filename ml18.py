
    Performance metrics for Classification Based Algorithms
    
    1. Confusion Matrix
    2. Accuracy
    3. Precision
    4. Recall
    5. F1 Score
    6. ROC Curve
    7. AUC Score
    8. Type I Error
    9. Type II Error
    
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

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

(50 + 47 + 49) / 150

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(y, y_pred)
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)














































    