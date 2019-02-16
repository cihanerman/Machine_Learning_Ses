#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:45:47 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Churn_Modelling.csv')

data = data.iloc[:,3:]
x = data.iloc[:,:10].values
y = data.iloc[:,10:].values

# data normalization
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
x = x[:,1:]

# data train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 0)

# data scaling
sc =  StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


from xgboost import XGBClassifier

xgb = XGBClassifier().fit(x_train,y_train)
y_pred_xgb = xgb.predict(x_test)
cm2 = confusion_matrix(y_test,y_pred_xgb)
print('xgboost: ')
print(cm2)
print(accuracy_score(y_test,y_pred_xgb))