#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:23:04 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
 
# data loading
data = pd.read_csv('veriler.csv')
x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

gnb = GaussianNB().fit(X_train,y_train)
predict_gnb = gnb.predict(X_test)

print('GaussianNB')
print(predict_gnb)
print(y_test)
print(y_test[0] == predict_gnb)
cm = confusion_matrix(y_test,predict_gnb)
print(cm)

mnb = MultinomialNB().fit(x_train,y_train) # negatif input almıyor
predict_mnb = mnb.predict(X_test)

print('MultinomialNB')
print(predict_mnb)
print(y_test)
print(y_test[0] == predict_mnb)
cm = confusion_matrix(y_test,predict_mnb)
print(cm)

# Bu veri seti için en başarılı naive bayes algoritması
cnb = ComplementNB().fit(x_train,y_train) # negatif input almıyor
predict_cnb = cnb.predict(X_test)

print('ComplementNB')
print(predict_cnb)
print(y_test)
print(y_test[0] == predict_cnb)
cm = confusion_matrix(y_test,predict_cnb)
print(cm)