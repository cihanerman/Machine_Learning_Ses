#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:50:57 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Churn_Modelling.csv')

data = data.iloc[:,3:]
x = data.iloc[:,:10].values
y = data.iloc[:,10:].values

# data normalization
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
x = x[:,1:]

# data train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state = 0)

# data scaling
sc =  StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# create deep learning model
import keras
from keras.models import Sequential #Yapay sinir ağı 
from keras.layers import Dense # gizli katman

model = Sequential() # Yapay sinir ağı tanımlama
model.add(Dense(6, init='uniform', activation='relu', input_dim=11)) # modele gizli katman ekleme
model.add(Dense(6, init='uniform', activation='relu')) # modele gizli katman ekleme # inpu_dim sadece ilk katmanda verilir
model.add(Dense(1, init='uniform', activation='sigmoid')) # çıkış katmanı
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # model derlemesi

# model fit and predict
model.fit(x_train,y_train, epochs=50)

y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print(cm)