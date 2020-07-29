
    
        Machine Learning Introductory Class
    
Spyder is an IDE for scientific computing.
It comes pre-bundled along with Anaconda Distribution.

Anaconda Distribution is a free and open source
software stack for scientific computing.

Machine Learning is an interdisciplinary domain that
combines statistics and computer science to develop
intelligent applications and products.

Some experts believe ML to be fancy statistics.

False Notions associated with ML

"The term Machine Learning was coined by Sir Arthur \
 Samuel of the Stanford University in the year 1959."
 
"Dr Samuel was HOD of the statistics department at \
 Stanford."
 
1. ML is a new branch. No it is not. It is new in a 
sense that commoners have started using it.


        SDLC (Software Development Lifecycle)
        
It is well defined series of steps to develop an
enterprise level software.

1. Requirement Analysis
2. Designing
3. Development
4. Testing
5. Implementation
6. Maintenance

ML is not software development.
But we have mentioned SDLC because we know that while
working in the industry we do need some guidelines
to follow.

        End to End Machine Learning Project
        
1. Get the Data
2. Discovery and Visualization to get Insights
3. Data Preprocessing
4. Select and Train an ML Algorithm
5. Fine Tuning
6. Launch, Maintain, and Monitor your System

        End to End Analytics Project
        
1. Get the Data
2. Descriptive Statistics
3. EDA (Exploratory Data Analysis) 
4. Estimation
5. Hypothesis Testing

Machine Learning Syllabus

1. Important ML Libraries (Numpy, Matplotlib etc)
2. Regression Based Algorithms
    a) Simple Linear Regression
    b) Polynomial Regression
3. Classification Based Algorithms
    a) KNN
    b) Logistic Regression
    c) SVM
    d) Decision Tree
    e) Naive Bayes
4. Clustering Based Algorithms
    a) K-Means Clustering
5. Dimensionality Reduction
6. Fine Tuning
7. Projects

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_list = list(range(1000000))

We have stored 10 lac elements indexed from 0 to
9,99,999 in a traditional python list.

We will now store the same 10 lac elements in a 
Numpy Array and check which one is faster.

my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2

This small program proves how Numpy is much much
faster than traditional python.
In ML, it is heavily recommended to use Numpy
data structures in place of normal python data 
structures.


















































        
        
        
        
        
        
        
        
        








































