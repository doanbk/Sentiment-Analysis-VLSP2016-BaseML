# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from pyvi.pyvi import ViPosTagger, ViTokenizer
import string

def readdata(path):
    data = []
    with open(path, 'r') as f:
        rawdata = f.read().splitlines()
    for onecomment in rawdata:
        data.append(onecomment.split(':', 1))
    X = [data[i][1] for i in range(len(data))]
    Y = [data[i][0] for i in range(len(data))]

    return X, Y

def clean_data(comment):

    comment = re.sub("(?s)<ref>.+?</ref>", "", comment)  # remove reference links
    comment = re.sub("(?s)<[^>]+>", "", comment)  # remove html tags
    comment = re.sub("&[a-z]+;", "", comment)  # remove html entities
    comment = re.sub("(?s){{.+?}}", "", comment)  # remove markup tags
    comment = re.sub("(?s){.+?}", "", comment)  # remove markup tags
    comment = re.sub("(?s)\[\[([^]]+\|)", "", comment)  # remove link target strings
    comment = re.sub("(?s)\[\[([^]]+\:.+?]])", "", comment)  # remove media links

    listpunctuation = string.punctuation

    for i in listpunctuation:
        comment = comment.replace(i, " ")

    comment = comment.lower()

    return comment

def convert_abbreviation(comment):
    filename = './data/dict/dict_abbreviation.csv'
    data = pd.read_csv(filename, sep="\t", encoding='utf-8')
    list_abbreviation = data['abbreviation']
    list_converts = data['convert']

    comment = comment.decode('utf-8')
    comment = re.sub('\s+', " ", comment)

    for i in range(len(list_converts)):
        abbreviation = '(\s' + list_abbreviation[i] + '\s)|(^' + list_abbreviation[i] + '\s)|(\s' \
                        + list_abbreviation[i] + '$)'
        convert = ' ' + list_converts[i] + ' '
        comment = re.sub(abbreviation, convert , comment)

    return comment

def tokenize(comment):
    list_token = []
    text_token = ViTokenizer.tokenize(comment)
    # ViPosTagger.postagging(text_token)
    return text_token

def stopword(comment):
    filename = './data/dict/stopwords.csv'
    data = pd.read_csv(filename, sep="\t", encoding='utf-8')
    list_stopwords = data['stopwords']
    re_commnent = []
    words = comment.split()
    for word in words:
        if word not in list_stopwords:
            re_commnent.append(word)
    comment = " ".join(re_commnent)

    return comment

def predata(path):
    X, Y = readdata(path)
    X_re = []
    for comment in X:
        comment = tokenize(stopword(convert_abbreviation(clean_data(comment))))
        X_re.append(comment)
    return X_re

def build_dict(X):
    _dict = []
    for comment in X:
        _dict.extend(comment.split())
    _dict = list(set(_dict))
    return _dict

X_re = predata('./data/train/train.txt')

list_sentiment = []

for comment in X_re:
    token = ViTokenizer.tokenize(comment)
    token_split = token.split()
    Pos = ViPosTagger.postagging(token)[1]

    for i in range(len(Pos) -1):
        if (Pos[i] == 'N' and Pos[i + 1] == "A"):
            list_sentiment.append(token_split[i + 1])
        if (Pos[i] == 'R' and Pos[i + 1] == "A"):
            list_sentiment.append(token_split[i + 1])
        if (Pos[i] == 'R' and Pos[i + 1] == "V"):
            list_sentiment.append(token_split[i + 1])
        if (Pos[i] == 'V' and Pos[i + 1] == "A"):
            list_sentiment.append(token_split[i + 1])
list_sentiment = set(list_sentiment)
with open('hahahaha.txt', 'w') as f:
    for text in list_sentiment:
        f.write(text.encode('utf-8'))
        f.write('\n')

# for text in list_sentiment:
#     print text