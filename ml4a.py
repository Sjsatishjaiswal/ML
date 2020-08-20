
            Regression Based Algorithms
            
1. Simple Linear Regression
2. Polynomial Regression
3. Regularized Regression

            Simple Linear Regression
            
Simple : Easy
Linear : Something that resembles a line or line itself
Regression : Practices or patterns of the past

The mathematics behind this algorithm was formulated by
Gauss in the year 1795 with the name of OLS. Later patented
by Legendre in the year 1805.

Assumption of SLR is that the data must be linearly 
distributed.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

1. Get the Data

dataset = pd.read_excel('blood.xlsx')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, 2].values
X = X.reshape(-1, 1)

2. Discovery and Visualization

plt.scatter(X, y)
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Analysis on the Blood Dataset')
plt.show()

3. Data Preprocessing

No need

4. Select and Train an ML Algo

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.predict([[20]])
lin_reg.predict([[30]])

c = lin_reg.intercept_

m = lin_reg.coef_

m, c

ayush = m * 20 + c
ayush

rahul = m * 30 + c
rahul

y_pred = lin_reg.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c = "red")
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Analysis on the Blood Dataset')
plt.show()

lin_reg.score(X, y)































































