#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:17:19 2019

@author: cihanerman
"""
"""
PCA ve LDA boyut indirgiyerek performans artışı sağlar
PCA: Sınıf farkı gözetmez
LDA: Sınıf farkı gözetir
"""
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import re
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Wine.csv')

x = data.iloc[:,0:13].values
y = data.iloc[:,13:].values

# data train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state = 0)

# data scaling
sc =  StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# PCA
pca = PCA(n_components=2)

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

model = LogisticRegression(random_state=0).fit(x_train,y_train)
y_pred = model.predict(x_test)

model2 = LogisticRegression(random_state=2).fit(x_train2,y_train)
y_pred2 = model2.predict(x_test2)

print("PCA: ")
cm1 = confusion_matrix(y_test,y_pred)
print(cm1)

cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

# LDA
lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda = lda.transform(x_test)

model_lda = LogisticRegression(random_state=0).fit(x_train_lda,y_train)
y_pred_lda = model_lda.predict(x_test_lda)

print("LDA: ")
cm_lda = confusion_matrix(y_pred,y_pred_lda)
print(cm_lda)
