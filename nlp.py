#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:16:28 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer # kelimelerin frekansını bulmada kullanılıyor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Restaurant_Reviews.csv')

# sparce matrix (Çoğunluğu boş matris)
# Preprocessing (Önişlme)
ps = PorterStemmer() # kelimelerin eklerini çıkarıp kökü dödürüyor.
data.Review = [' '.join([ps.stem(y) for y in re.sub('[^a-zA-Z]',' ',x.lower()).split() if y not in set(stopwords.words('english'))]) for x in data.Review]

#stopwords = nltk.download('stopwords')
# Feature Extraction (Öz nitelik çıkarımı)
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(data.Review).toarray()
y = data.Liked.values

# data train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 0)

# model create, fit and predict
gnb = GaussianNB().fit(x_train,y_train)
predict_gnb = gnb.predict(x_test)

print('GaussianNB')
cm = confusion_matrix(y_test,predict_gnb)
print(cm)
print(accuracy_score(y_test,predict_gnb))

