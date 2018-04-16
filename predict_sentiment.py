# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from clean_data import *

clf = joblib.load('model.pkl')
vectorizer = joblib.load('vectorembedding.pkl')

comment = 'nó đẹp'
comment = remove_Stopword(tokenize(convert_Abbreviation(normalize_Text(clean_data(comment)))))

listcomment = []
listcomment.append(comment)
vectorcomment = vectorizer.transform(listcomment).toarray()

print clf.predict(vectorcomment)