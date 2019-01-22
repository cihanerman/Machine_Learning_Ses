#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 21:36:40 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski').fit(X_train,y_train)
"""
n_neighbors değeri veriye göre seçilmelidir. Bu veri setinde düşürmek daha doğru sonuçlar veriyor. Default değeri 5 dir.
metric değeri oluşturulacak modele göre seçilmelidir. Default değeri minkowski dir.
"""
print(knn.predict(X_test))
print(y_test[0])
print(y_test[0] == knn.predict(X_test))
cm = confusion_matrix(y_test,knn.predict(X_test))
print(cm)