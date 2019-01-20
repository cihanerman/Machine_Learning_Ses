#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 21:08:23 2019

@author: cihanerman
"""


# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# data loading
data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

# linear regression 
lr = LinearRegression().fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lr.predict(x), color = 'blue')
plt.show()
print('r2 score: ',r2_score(y,lr.predict(x)))
# plynomal regression degree = 2
pr = PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression().fit(x_poly,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lr2.predict(x_poly), color = 'blue')
plt.show()

# plynomal regression degree = 4
pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression().fit(x_poly,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lr2.predict(x_poly), color = 'blue')
plt.show()

# predicts
print(lr.predict([[11],[6.6]]))
print(lr2.predict(pr.fit_transform([[11],[6.6]])))

print('r2 score: ',r2_score(y,lr2.predict(pr.fit_transform(x))))