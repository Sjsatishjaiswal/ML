
    
    We can exploit the mathematical property of 
    OLS associated with the 'lop' to our advantage.
    
    In simple terms, there are two common techniques to
    deduce the coefficients of line of prediction.
    Which are:
        
        1. Normal Equation
        2. Gradient Descent
        
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = 100
X = np.random.randn(m, 1)
y = 5 * X + 7

plt.plot(X, y)
plt.show()

m = 100
X = np.random.randn(m, 1)
y = 14 * X + 45 + np.random.randn(m, 1)

plt.scatter(X, y)
plt.show()

X = np.c_[np.ones(m), X]

theta = np.linalg.inv(X.T @ X) @ X.T @ y
theta




































