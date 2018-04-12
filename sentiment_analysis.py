# -*- coding: utf-8 -*-

import vectorize_data as vd

X_train, y_train, X_test, y_test = vd.tf_Idf('./data/train/pre_train.txt','./data/test/pre_test.txt')

#scikit_learn: Linear SVM
def linear_SVM(c):
    from sklearn.svm import SVC

    clf = SVC(kernel="linear", C=c)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
# print 'linear_SVM: ',linear_SVM(0.9) #67.8%

# Nearest Neighbors
def nearest_Neighbors(n):
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score
# print nearest_Neighbors(9) #62%

# Decision Tree
def decision_Tree(n):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=n)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# print decision_Tree(31) #52%

# Random Forest
def random_Forest(depth = 10, estimator = 20):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=depth, n_estimators= estimator, max_features=20)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
# print 'random_Forest: ',random_Forest(depth=35) #57%


# Neural Net
def neural_Net(alpha):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(alpha=alpha)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
# print 'neural_Net: ',neural_Net(1) #69% #layer: default

# Naive Bayes
def naive_Bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
# print 'naive_Bayes: ',naive_Bayes() #53%

#Maxent
def logictic(c):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', max_iter=100, C=c)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

# print 'logictic: ',logictic(0.7) #68