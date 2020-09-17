
        Support Vector Machine (SVM)
        
Support : Sahara (Something that supports your claim)
Vector : 1D mathematical structure
Machine : 3 idiots

SVMs can only work with linearly separable datasets.
For non-linearly separably datasets, we use a different
flavor of SVM known as Kernel-SVM.

It is a binary classifier. For n-ary cases it uses
the OVO/OVA trick.

It can work with small and simple datasets.
Not suitable for moderate or large datasets.

from sklearn.svm import SVC
svm = SVC()