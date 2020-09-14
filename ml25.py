
##### Handwritten Digits Image Classification Model ##########

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

X = dataset.data
y = dataset.target
y = y.astype(int)

X, y

some_digit = X[56789]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image)
plt.show()

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    img = X[i]
    img = img.reshape(28, 28)
    plt.imshow(img)
    plt.xlabel(y[i])
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    img = X[i]
    img = img.reshape(28, 28)
    plt.imshow(img)
    plt.xlabel('Actual : {} \nPredicted : {}'.format(y[i], y_pred[i]))
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)

y_pred_test = log_reg.predict(X_test)

for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    img = X_test[i]
    img = img.reshape(28, 28)
    plt.imshow(img, "binary")
    plt.xlabel('Actual : {} \nPredicted : {}'.format(y_test[i], y_pred_test[i]))
plt.tight_layout()
plt.show()































