# -*- coding: utf-8 -*-

file_neu = open("./data/neutral.txt")
data_neu = file_neu.read().split('\n\n')
m1 = len(data_neu)
file_neu.close()

file_neg = open("./data/negative.txt")
data_neg = file_neg.read().split('\n\n')
m2 = len(data_neg)
file_neg.close()

file_pos = open("./data/positive.txt")
data_pos = file_pos.read().split('\n\n')
m3 = len(data_pos)
file_pos.close()

data = data_neu + data_neg + data_pos

from sklearn.model_selection import train_test_split

train1, test1 = train_test_split(data_neu, test_size=340)
train2, test2 = train_test_split(data_neg, test_size=340)
train3, test3 = train_test_split(data_pos, test_size=340)

with open('./data/train/train.txt', 'w') as f:
    for text in train1:
        f.write('0: '+text + '\n')
    for text in train2:
        f.write('1: '+text + '\n')
    for text in train3:
        f.write('2: '+text + '\n')

with open('./data/test/test.txt', 'w') as f:
    for text in test1:
        f.write('0: '+text + '\n')
    for text in test2:
        f.write('1: '+text + '\n')
    for text in test3:
        f.write('2: '+text + '\n')