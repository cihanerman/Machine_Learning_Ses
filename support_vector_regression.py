#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:20:55 2019

@author: cihanerman
"""
# svg marjinal değerlere karşı dayanıksız
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# data loading
data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

# standadization
sc1 = StandardScaler()
x_train = sc1.fit_transform(x)
sc2 = StandardScaler()
y_test = sc2.fit_transform(y)

# svg models
sr_r = SVR(kernel='rbf').fit(x_train,y_test)
sr_l = SVR(kernel='linear').fit(x_train,y_test)
sr_p = SVR(kernel='poly').fit(x_train,y_test)

# plottting
plt.scatter(x_train,y_test, color='orange')
plt.plot(x_train,sr_r.predict(x_train), color='red')
plt.plot(x_train,sr_l.predict(x_train), color='blue')
plt.plot(x_train,sr_p.predict(x_train), color='green')
plt.show()

# predicts
print(sr_r.predict([[11]]))
print(sr_r.predict([[6.6]]))
print('sr_r r2 score: ',r2_score(y_test,sr_r.predict(x_train)))

print(sr_l.predict([[11]]))
print(sr_l.predict([[6.6]]))
print('sr_l r2 score: ',r2_score(y_test,sr_l.predict(x_train)))

print(sr_p.predict([[11]]))
print(sr_p.predict([[6.6]]))
print('sr_p r2 score: ',r2_score(y_test,sr_p.predict(x_train)))