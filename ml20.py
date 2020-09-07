import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col_names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'gender', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country',
             'salary']


dataset = pd.read_csv('adult_salary.csv', names = col_names,
                      na_values = ' ?')

dataset.isnull().sum()

dataset_int = dataset.iloc[:, [0, 2, 4, 10, 11, 12]]
dataset_str = dataset.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13]]

from sklearn.impute import SimpleImputer
sim = SimpleImputer(strategy = 'most_frequent') 
dataset_str = sim.fit_transform(dataset_str)

dataset_str = pd.DataFrame(dataset_str)
dataset_str.isnull().sum()

dataset_str = pd.get_dummies(dataset_str)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset_int = scaler.fit_transform(dataset_int)

dataset_int = pd.DataFrame(dataset_int)

dataset_final = pd.concat([dataset_int, dataset_str], axis = 1)

dataset_final.isnull().sum()

X = dataset_final.values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
lab.classes_

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)












































