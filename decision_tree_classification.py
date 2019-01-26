#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:05:03 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

dt = DecisionTreeClassifier().fit(X_train, y_train)
predict = dt.predict(X_test)

print('gini')
print(predict)
print(y_test)
print(y_test[0] == predict)
cm = confusion_matrix(y_test,predict)
print(cm)

dt_e = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
predict_e = dt_e.predict(X_test)

print('entropy')
print(predict)
print(y_test)
print(y_test[0] == predict_e)
cm = confusion_matrix(y_test,predict_e)
print(cm)