# -*- coding: utf-8 -*-

import vectorize_data as vd
from sklearn.externals import joblib

X_train, y_train, X_test, y_test = vd.tf_Idf('./data/train/pre_train.txt','./data/test/pre_test.txt')

#scikit_learn: SVM
def SVM():
    from sklearn.svm import SVC

    clf = SVC(kernel="linear", C=0.1)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    joblib.dump(clf, 'model.pkl')

    return score
# print 'SVM: ', SVM()

# Neural Net
def neural_Net(alpha):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(alpha=alpha)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    joblib.dump(clf, 'model.pkl')

    return score
# print 'neural_Net: ',neural_Net(1)
# neural_Net 20-1:  0.688235294118
# neural_Net 100-1: 69.9%


#Maxent
def logictic(c):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', max_iter=100, C=c)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    joblib.dump(clf, 'model.pkl')

    return score

print 'logictic: ',logictic(10)