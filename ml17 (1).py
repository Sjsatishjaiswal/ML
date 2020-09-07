
    
                Logistic Regression
                
The name of the algorithm is a misnomer. It is actually
a classification algorithm but its name is regression.

The simplest type of classification is binary 
classification.

Logistic : Counting or Measuring or Estimating
Regression : Woh aapko pata he hai

If we use linear coefficients for the binary 
classification problem then there exists some value of
'x' for which we will get wrong or ambiguous 
probabilistic outcome.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(-10, 10, 0.01)
y = 1 / (1 + np.power(np.e, -x))
y1 = np.power(np.e, -x) / (1 + np.power(np.e, -x))

plt.plot(x, y)
plt.show()

plt.plot(x, y1)
plt.show()





















