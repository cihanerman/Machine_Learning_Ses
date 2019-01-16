#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:17:15 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# data loading
data = pd.read_csv('odev_tenis.csv')
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features='all')
data.windy = le.fit_transform(data.windy)
data.play = le.fit_transform(data.play)
data.outlook = le.fit_transform(data.outlook)

outlook = ohe.fit_transform(data.iloc[:,0:1].values).toarray()
df = pd.DataFrame(data = outlook,index = range(14), columns = ['o','r','s'])
data2 = pd.concat([df,data.iloc[:,1:]], axis = 1)
drop1 = data2.drop(['humidity'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(drop1,data2.humidity, test_size=0.33, random_state = 0)
lr = LinearRegression().fit(x_train, y_train)
predict = lr.predict(x_test)

X = np.append(arr = np.ones((14,1)).astype(int), values = drop1, axis = 1 )
x_ls = drop1.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog = data2.humidity, exog = x_ls).fit()
print(r.summary())

drop1 = drop1.drop(['windy'], axis=1)

X = np.append(arr = np.ones((14,1)).astype(int), values = drop1, axis = 1 )
x_ls = drop1.iloc[:,[0,1,2,3,4]].values
r = sm.OLS(endog = data2.humidity, exog = x_ls).fit()
print(r.summary())

x_train = x_train.drop(['windy'], axis=1)
x_test = x_test.drop(['windy'], axis=1)
lr.fit(x_train, y_train)
predict2 = lr.predict(x_test)
