#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:34:24 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
 
# data loading
data = pd.read_csv('maaslar_yeni.csv')

x = data.iloc[:,2:3].values
y = data.iloc[:,5:].values

# seaborn with corrrlations
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# linear regression model create
lr = LinearRegression().fit(x, y)

# p value
"""
p value değeri verideki hangi sütünların bizim modelimizi tahminini ne kadar
etkili olacağını belirlememiz de kullanılır. 0'a yakın olan sütunlar tercih
edilir. 
"""
sm_model = sm.OLS(lr.predict(x),x)
print(sm_model.fit().summary()) # bunu sonucu sadece il sütün değerleri alınır.

# plynomal regression degree = 2
pr = PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression().fit(x_poly,y)

sm_model2 = sm.OLS(lr2.predict(pr.fit_transform(x)),x)
print(sm_model2.fit().summary())

# standadization
sc1 = StandardScaler()
x_train = sc1.fit_transform(x)
sc2 = StandardScaler()
y_test = sc2.fit_transform(y)

# svg models
sr_r = SVR(kernel='rbf').fit(x_train,y_test)
sm_model3 = sm.OLS(sr_r.predict(x_train),x_train)
print(sm_model3.fit().summary())

# decision tree create model
dt_r = DecisionTreeRegressor(random_state=0).fit(x,y)
sm_model4 = sm.OLS(dt_r.predict(x),x)
print(sm_model4.fit().summary())

# random forest model create
rf = RandomForestRegressor(n_estimators=10,random_state=0).fit(x,y)
sm_model5 = sm.OLS(rf.predict(x),x)
print(sm_model5.fit().summary())