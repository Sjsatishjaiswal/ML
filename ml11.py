import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing.csv')
dataset.isnull().sum()

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 8].values

from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values = np.nan, strategy = 'median')
sim.fit(X[:, [4]])
sim.statistics_
X[:, [4]] = sim.transform(X[:, [4]])
# sim.fit_transform()

##################### Label Encoding ####################

Gender
male
female
female
others
male

After Label Encoding

Gender
0
1
1
2
0

Here the process of label encoding has given rise to a new problem
known as dummy variable trap. 

Dummy Variable Trap : Since the arithmetic value of 2 > 1 > 0
some stupid ML algorithm might think that the priority of 'others'
is greater than 'female' which is again greater than 'male'.
Which would be an incorrect assumption to make as the gender column
had inherently had 'categorical' values which have no order
in priority. This problem is famously known as DVT.

The solution to this problem is known as Dummy Variable Encoding.
Which simply means creating the sparse matrix representation of
the label encoded column.

{male : 0 | female : 1 | others : 2} ----> Label Encoding

Gender
0
1
1
2
0                          

after DVE,

0   1   2
1   0   0
0   1   0
0   1   0
0   0   1
1   0   0

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 8] = lab.fit_transform(X[:, 8])
lab.classes_

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.score(X, y)

y_pred = lin_reg.predict(X)




























































