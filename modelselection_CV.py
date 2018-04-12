# -*- coding: utf-8 -*-

import vectorize_data as vd

X_train, y_train, X_test, y_test = vd.tf_Idf('./data/train/pre_train.txt', './data/test/pre_test.txt')


# Maxent
def logictic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    lr = LogisticRegression()

    param_grid = {'penalty': ['l1', 'l2'], 'max_iter': [100], 'C': [1, 10, 50, 100]}
    clf = GridSearchCV(lr, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score
# logictic:  (0.67009803921568623, {'penalty': 'l2', 'C': 10, 'max_iter': 100}, 0.69411764705882351)


print 'logictic: ', logictic()


# Nearest Neighbors
def nearest_Neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    clf = GridSearchCV(knn, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)
    return best_score, best_param, score
# nearest_Neighbors:  (0.61078431372549025, {'n_neighbors': 9}, 0.61078431372549025)


print 'nearest_Neighbors: ', nearest_Neighbors()


# Decision Tree
def decision_Tree():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    dt = DecisionTreeClassifier()

    param_grid = {'max_depth': [5, 10, 20, 30]}
    clf = GridSearchCV(dt, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score


print 'decision_Tree: ', decision_Tree()
# decision_Tree:  (0.51593137254901966, {'max_depth': 30}, 0.52549019607843139)


# scikit_learn: SVM
def SVM():
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    svc = SVC()

    param_grid = [
        {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4], 'kernel': ['poly']},
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    clf = GridSearchCV(svc, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score


print 'SVM: ', SVM()


# Naive Bayes
def naive_Bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score


print 'naive_Bayes: ', naive_Bayes()
# print 'naive_Bayes: ',naive_Bayes() #53%


# Random Forest
def random_Forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    rfc = RandomForestClassifier()

    param_grid = {'max_depth': [5, 10], 'n_estimators': [5, 10, 20], 'max_features': [10, 20]}
    clf = GridSearchCV(rfc, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score


print 'random_Forest: ', random_Forest()


# Neural Net
def neural_Net(alpha):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    mlp = MLPClassifier()

    param_grid = {'alpha': [0.1, 1], 'hidden_layer_sizes': [100, 20]}
    clf = GridSearchCV(mlp, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score
# print 'neural_Net: ',neural_Net(1) #69% #layer: default
