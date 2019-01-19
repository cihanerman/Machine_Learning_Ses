#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:56:27 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# data loading
data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

rf = RandomForestRegressor(n_estimators=10,random_state=0).fit(x,y)

plt.scatter(x,y, color='orange')
plt.plot(x, rf.predict(x), color='gray')
plt.show()

print(rf.predict([[11]]))
print(rf.predict([[6.6]]))