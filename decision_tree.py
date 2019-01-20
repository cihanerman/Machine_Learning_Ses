#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:19:09 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# data loading
data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

# decision tree create model
dt_r = DecisionTreeRegressor(random_state=0).fit(x,y)

plt.scatter(x,y, color='orange')
plt.plot(x, dt_r.predict(x), color='gray')
plt.show()

print(dt_r.predict([[11]]))
print(dt_r.predict([[6.6]]))

print('r2 score: ',r2_score(y,dt_r.predict(x)))