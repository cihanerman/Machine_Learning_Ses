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
from sklearn.metrics import r2_score

# data loading
data = pd.read_csv('maaslar.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

# random forest model create
rf = RandomForestRegressor(n_estimators=10,random_state=0).fit(x,y)

plt.scatter(x,y, color='orange')
plt.plot(x, rf.predict(x), color='gray')
plt.show()

print(rf.predict([[11]]))
print(rf.predict([[6.6]]))

print('r2 score: ',r2_score(y,rf.predict(x)))
# r squer algoritmaları karşılaştırmak için bir yol.