#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:28:57 2019

@author: cihanerman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
 
# data loading
data = pd.read_excel('Iris.xls')

x = data.iloc[:,:4]
y = data.iloc[:,4:]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state = 0)

# logistic regression
lrc = LogisticRegression().fit(x_train,y_train)
predict = lrc.predict(x_test)
print('lrc accuracy: ',accuracy_score(y_test,predict))
print('lrc cm:', confusion_matrix(y_test,predict))

# knn
knn = KNeighborsClassifier().fit(x_train,y_train)
predict = knn.predict(x_test)
print('knn accuracy: ',accuracy_score(y_test,predict))
print('knn cm:', confusion_matrix(y_test,predict))

# SVM
svm = SVC(kernel='rbf').fit(x_train,y_train)
predict = svm.predict(x_test)
print('svm accuracy: ',accuracy_score(y_test,predict))
print('svm cm:', confusion_matrix(y_test,predict))

# native bayes
gnb = GaussianNB().fit(x_train,y_train)
predict = gnb.predict(x_test)
print('gnb accuracy: ',accuracy_score(y_test,predict))
print('gnb cm:', confusion_matrix(y_test,predict))

# desiciom tree
dtc = DecisionTreeClassifier().fit(x_train,y_train)
predict = dtc.predict(x_test)
print('dtc accuracy: ',accuracy_score(y_test,predict))
print('dtc cm:', confusion_matrix(y_test,predict))

# desiciom tree
rfc = RandomForestClassifier().fit(x_train,y_train)
predict = rfc.predict(x_test)
print('dtc accuracy: ',accuracy_score(y_test,predict))
print('dtc cm:', confusion_matrix(y_test,predict))