# -*- coding: utf-8 -*-
import vectorize_data as vd


X_train, y_train, X_test, y_test = vd.Bow('./data/train/pre_train.txt','./data/test/pre_test.txt')

#scikit_learn: Linear SVM
def linear_SVM():
    from sklearn.svm import SVC

    clf = SVC(kernel="linear", C=0.025)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
print linear_SVM()

# SVM
def svm():
    from sklearn.svm import SVC
    clf = SVC(gamma=2, C=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# Nearest Neighbors
def nearest_Neighbors():
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# Decision Tree
def decision_Tree():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# Random Forest
def random_Forest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# Neural Net
def neural_Net():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(alpha=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# Naive Bayes
def naive_Bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

def MLNN():
    #learning rate, l2_regular
    #stochastic Gradien Descent + early stopping + 20 epochs
    pass

def logictic():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=0.001)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score