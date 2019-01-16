#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:20:49 2019

@author: cihanerman
"""
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# data loading
data = pd.read_csv('veriler.csv')

contry = data.iloc[:,0:1].values
print(contry)
le = LabelEncoder() # labelları bire bir sayıya çeviriyor
contry[:,0] = le.fit_transform(contry[:,0])
print(contry)
ohe = OneHotEncoder(categorical_features='all') # kolon başlığına çeviriyor
contry = ohe.fit_transform(contry).toarray()

gender = data.iloc[:,-1:].values
print(gender)
le = LabelEncoder() # labelları bire bir sayıya çeviriyor
gender[:,0] = le.fit_transform(gender[:,0])
print(gender)
ohe = OneHotEncoder(categorical_features='all') # kolon başlığına çeviriyor
gender = ohe.fit_transform(gender).toarray()

other = data.iloc[:,1:4].values

s1 = pd.DataFrame(data = contry, index = range(22), columns = ['fr','tr','us'])
s2 = pd.DataFrame(data = other, index = range(22), columns = ['height','weight','age'])
s3 = pd.DataFrame(data = gender[:,:1], index = range(22), columns = ['gender'])
print(s1)
print(s2)
print(s3)
s = pd.concat([s1,s2], axis = 1)
st = pd.concat([s1,s2,s3], axis = 1)
print(s)

x_train, x_test, y_train, y_test = train_test_split(s,s3, test_size=0.33, random_state = 0)

lr = LinearRegression().fit(x_train, y_train)
gender_predict = lr.predict(x_test)

height = st.iloc[:,3:4]
left = st.iloc[:,:3]
right = st.iloc[:,4:]
sn = pd.concat([left,right], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(sn,height, test_size=0.33, random_state = 0)
lr2 = LinearRegression().fit(x_train, y_train)
height_predict = lr2.predict(x_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values = st, axis = 1 )
x_ls = st.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog = height, exog = x_ls).fit()
print(r.summary())