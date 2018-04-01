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
# print linear_SVM() #68%
# print 'linear_SVM: ',linear_SVM(0.9) #67%

# SVM
def svm():
    from sklearn.svm import SVC
    clf = SVC(gamma=2, C=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

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

# print 'decision_Tree: ',decision_Tree(20) #52%
# print 'decision_Tree: ',decision_Tree(25) #52%

# Random Forest
def random_Forest(depth = 10, estimator = 20):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=depth, n_estimators= estimator, max_features=20)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
# print random_Forest(depth=30) #57%
# print 'random_Forest: ',random_Forest(depth=30) #57%
# print 'random_Forest: ',random_Forest(depth=35) #57%


# Neural Net
def neural_Net(alpha):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(alpha=alpha)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

print neural_Net(1) #69% #layer: default

# Naive Bayes
def naive_Bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

# print naive_Bayes() #49%
# print 'naive_Bayes: ',naive_Bayes() #49%

def logictic(c):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C = c)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

# print logictic() #63

# print 'logictic: ',logictic(0.5) #67
# print 'logictic: ',logictic(0.7) #67