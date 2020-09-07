
    
    ROC (Receiver Operating Characteristic) Curve
    
A truly random classifier will have some correct
predictions by the virtue of probability.

The ROC Curve is used to deduce the relationship between
a truly random classifier and your algorithmic ML model.

The ROC Curve plots the 'true positive rate' (TPR)
against the 'false positive rate' (FPR).

True Positive Rate (TPR) is another name for Recall.

False Positive Rate (FPR) is the ratio of negative instances
that are incorrectly classified as positive.

FPR = 1 - TNR (True Negative Rate)

True Negative Rate (TNR) is the ratio of negative 
instances that are correctly classified as negative.

In english,

Recall = Sensitivity
TNR = Specificity

"ROC Curve plots sensitivity against 1 - specificity."

Sometimes it gets difficult to visualize outcome
in the ROC Curve as two or more than two curves can 
be very close to each other.

To resolve this problem, we can computed a new trick
known as AUC Score (Area Under the Curve) which 
ranges between [0 to 1].

The higher the AUC Score, the better the model is.

"The AUC Score of a truly random classifier is 0.5"











































