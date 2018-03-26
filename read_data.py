# -*- coding: utf-8 -*-

import re
from ctypes import c_bool

import pandas as pd
from pyvi.pyvi import ViTokenizer
import string
from vnaccent import accent

# data dict_abbreviation
filename = './data/dict/dict_abbreviation.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_abbreviation = data['abbreviation']
list_converts = data['convert']

# data stopwords
filename = './data/dict/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']


def readdata(path):
    data = []
    with open(path, 'r') as f:
        rawdata = f.read().splitlines()
    for onecomment in rawdata:
        data.append(onecomment.split(':', 1))
    X = [data[i][1] for i in range(len(data))]
    Y = [data[i][0] for i in range(len(data))]

    return X, Y

# loai spam va the html du
def clean_data(comment):
    comment = re.sub("(?s)<ref>.+?</ref>", "", comment)  # remove reference links
    comment = re.sub("(?s)<[^>]+>", "", comment)  # remove html tags
    comment = re.sub("&[a-z]+;", "", comment)  # remove html entities
    comment = re.sub("(?s){{.+?}}", "", comment)  # remove markup tags
    comment = re.sub("(?s){.+?}", "", comment)  # remove markup tags
    comment = re.sub("(?s)\[\[([^]]+\|)", "", comment)  # remove link target strings
    comment = re.sub("(?s)\[\[([^]]+\:.+?]])", "", comment)  # remove media links

    return comment


def normalize_Text(comment):
    comment = comment.decode('utf-8')
    comment = comment.lower()

    #thay gia tien bang text
    moneytag = ['k', 'đ', 'ngàn', 'nghìn', 'usd', 'tr', 'củ', 'triệu', 'yên']
    for money in moneytag:
        comment = re.sub('(^\d*[,.]?\d+\s*' + money + ')|(' + '\s\d*[,.]?\d+\s*' + money + ')', ' monney ', comment)
    comment = re.sub('(^\d+\s*\$)|(\s\d+\s*\$)', ' monney ', comment)
    comment = re.sub('(^\$\d+\s*)|(\s\$\d+\s*\$)', ' monney ', comment)
    comment = re.sub('(^\d+([.,]\d*)*\s*đ)|(\s\d+([.,]\d*)*\s*đ)', ' monney ', comment)

    #thay thong so bang text
    # comment = re.sub('^\d+\s*|\s\d+\s*', ' specifications ', comment)

    #xu ly lay am tiet
    comment = re.sub(r'(\D)\1+', r'\1', comment)

    # loai dau cau: nhuoc diem bi vo cau truc: vd; km/h. V-NAND
    listpunctuation = string.punctuation
    for i in listpunctuation:
        comment = comment.replace(i, ' ')


    # #them dau cho nhung cau khong dau
    # words = comment.split()
    # for word in words:
    #     try:
    #         comment.encode('utf-8')
    #         word = accent.accent_comment(word)
    #         print word
    #     except UnicodeError:
    #         word = word.decode('utf-8')
    #     words_normal.append(word)

    # # them dau tach thanh tung cau
    # sents = re.split("([.?!])?[\n]+|[.?!] ", comment)
    # listpunctuation = string.punctuation
    # sents_normal = []
    # # them dau theo cau
    # for sent in sents:
    #     if sent != None:
    #         for i in listpunctuation:
    #             sent = sent.replace(i, " ")
    #         sent = accent.accent_comment(sent.decode('utf-8'))
    #         sents_normal.append(sent.lower())
    return comment


def convert_Abbreviation(comment):
    comment = re.sub('\s+', " ", comment)
    for i in range(len(list_converts)):
        abbreviation = '(\s' + list_abbreviation[i] + '\s)|(^' + list_abbreviation[i] + '\s)|(\s' \
                       + list_abbreviation[i] + '$)'
        convert = ' ' + list_converts[i] + ' '
        comment = re.sub(abbreviation, convert, comment)

    return comment


def remove_Stopword(comment):
    re_commnent = []
    words = comment.split()
    for word in words:
        if word not in list_stopwords and word.isalpha():
            re_commnent.append(word)
    comment = ' '.join(re_commnent)

    return comment


def tokenize(comment):
    text_token = ViTokenizer.tokenize(comment)
    return text_token


def predata(path):
    X, Y = readdata(path)
    X_re = []
    i = 0
    for comment in X:
        comment = tokenize(convert_Abbreviation(normalize_Text(clean_data(comment))))
        X_re.append(comment)
        print i
        i += 1
    return X_re, Y


def build_Dict(X):
    dictionary = []
    for comment in X:
        dictionary.extend(comment.split())
    dictionary = list(set(dictionary))
    return dictionary


if __name__ == '__main__':
    X_re, Y = predata('./data/test/test.txt')
    dictionary = build_Dict(X_re)

    writer1 = open('./data/test/pre_test.txt', 'w')
    writer2 = open('./data/test/dict.txt', 'w')

    for i in range(len(X_re)):
        writer1.write(str(Y[i]) + ': ')
        writer1.write(X_re[i].encode('utf-8'))
        writer1.write('\n')

    for word in dictionary:
        writer2.write(word.encode('utf-8'))
        writer2.write('\n')