#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:56:00 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv('satislar.csv')
months = data[['Aylar']]
sales = data[['Satislar']] # dataframe data.Satislar Series oluyor.

# data splite train and test
x_train, x_test, y_train, y_test = train_test_split(months,sales, test_size=0.33, random_state = 0)
"""
# standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
# create modal (linear regression)
lr = LinearRegression().fit(x_train, y_train)
#print('score: ',lr.score(x_test,y_test))
predict = lr.predict(x_test)

# data sort
x_train = x_train.sort_index()
y_train = y_train.sort_index()

# data visualization
plt.plot(x_train,y_train)
plt.plot(x_test,predict)
plt.title('Aylara göre satış')
plt.xlabel('Aylar')
plt.ylabel('Satış')
plt.show()
