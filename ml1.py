
    
                    Kernel - SVM
                    
Kernel SVM is an improved implementation of vanilla
SVM to work on non-linearly separable datasets.

A kernel is simply a function that transforms an
'n' dimensional non-linearly separable dataset into an
'n+1' dimensional linearly separable dataset.

Once the dataset is linearly separable we can simply
apply SVM.

Some popular kernels are:
    1. RBF (Radial Basis Function)
    2. Polynomial
    3. Etc
    
The rbf kernel is based on a similarity function 
known as gaussian radial basis function.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.svm import SVC
svm = SVC()
svm.fit(X, y)

svm.score(X, y)

######################################################

            Natural Language Processing
            
NLP is a subset of computer science and machine learning
that attempts to derive meaning from textual data
and can help in the problems related to sentiment
analysis and chatbot creation etc.

The general steps for NLP are:
    1. Remove all the punctuations, numbers, symbols, emojis
        and unwanted characters.
    2. Get all the data in lower case.
    3. Remove unwanted words like preprositions, conjunctions
        determiners, fillers, pronouns etc.
    4. Perform stemming or lemmatization.
    5. Represent the data using an nlp model.

Restaurant Reviews

The food here was totally amazing...Like really really good!
100% recommended *jeebh nikalkar smile karne wala emoji*!!!!

The pizza was stale and the manager was rude. I waited 40
minutes for a piece of shit!! Disgusting...

1. The food here was totally amazing Like really really good
recommended 

2. the food here was totally amazing like really really good
recommended

3. food totally amazing like really good recommended

I love the food
I'm Loving it
I loved the food

{ love, loving, loved } --> Positive Predictor
{ love } --> Positive Predictor

4. food total amaz like real good recommend

5. Applying bag of words model

food : 0
total : 1
amaz : 2 
like : 3
real : 4
good : 5
recommend : 6

"food was good! Appreciated"
food good appreciate
0     5      12

0   1   2   3   4   5   6   7   8   9   10  11  12  13  14.....
1   1   1   1   1   1   1   0   0   0   0   0   0   0   0.....
1   0   0   0   0   1   0   0   0   0   0   0   1   0   0.....

#############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords

dataset = pd.read_csv('review.tsv', delimiter = '\t')

dataset['Review'][0]

clean_reviws = []

for i in range(1000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    text = text.lower()
    text = text.split()
    # t1 = [word for word in text if not word in set(stopwords.words('english'))]
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    clean_reviws.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 800)
X = cv.fit_transform(clean_reviws)
X = X.toarray()
y = dataset['Liked'].values

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

log_reg.score(X, y)

print(cv.get_feature_names())

Real Values for first 5 Reviews

y[:5]

[1, 0, 0, 1, 1]

y_pred = log_reg.predict(X)

y_pred[:5]
















































































































    

















































