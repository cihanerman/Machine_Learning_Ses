#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:01:58 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# data loading
data = pd.read_csv('veriler.csv')
contry = data.iloc[:,0:1].values
print(contry)
le = LabelEncoder() # labelları bire bir sayıya çeviriyor
contry[:,0] = le.fit_transform(contry[:,0])
print(contry)

ohe = OneHotEncoder(categorical_features='all') # kolon başlığına çeviriyor
contry = ohe.fit_transform(contry).toarray()
print(contry)

data2 = pd.read_csv('eksikveriler.csv')
# print(data2)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
age = data2.iloc[:,1:4].values
# print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)

s1 = pd.DataFrame(data = contry, index = range(22), columns = ['fr','tr','us'])
s2 = pd.DataFrame(data = age, index = range(22), columns = ['height','weight','age'])
s3 = pd.DataFrame(data = data.iloc[:,-1:].values, index = range(22), columns = ['gender'])
print(s1)
print(s2)
print(s3)
s = pd.concat([s1,s2], axis = 1)
print(s)

x_train, x_test, y_train, y_test = train_test_split(s,s3, test_size=0.33, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#%%
