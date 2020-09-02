
    We will discuss feature scaling today which is
    an important part of data preprocessing just like
    handling missing values and label encoding etc
    
    Sometimes, various features in the dataset have
    very different scales. This makes some ML biased
    towards the features with a larger scale. This
    is lead to problems in the prediction system
    which is why we need feature scaling i.e. to get
    all the features of the feature matrix in a same
    scale.
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_pre.csv')

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values = np.nan,
                    strategy = 'median')
X[:, 0:2] = sim.fit_transform(X[:, 0:2])

X_hot = pd.DataFrame(X[:, 2])
X_hot = pd.get_dummies(X_hot)
X_hot = X_hot.values

X = X[:, 0:2]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = np.concatenate([X, X_hot], axis = 1)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
lab.classes_

X, y

























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    