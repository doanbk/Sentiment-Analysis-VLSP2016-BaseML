# -*- coding: utf-8 -*-

import read_data as rd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def Bow(path_train, path_test):
    corpus_train, Y_train = rd.readdata(path_train)
    corpus_test, Y_test = rd.readdata(path_test)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()

    # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern = r'\b\w+\b', min_df = 1)
    # X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
    return X_train, Y_train, X_test, Y_test

def tf_Idf(path_train, path_test):
    corpus_train, Y_train = rd.readdata(path_train)
    corpus_test, Y_test = rd.readdata(path_test)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()
    return X_train, Y_train, X_test, Y_test
