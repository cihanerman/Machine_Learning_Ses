#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:45:48 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('musteriler.csv')

x = data.iloc[:,3:]

kmeans = KMeans(n_clusters=3,init='k-means++',random_state=0).fit(x)
print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0).fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11), sonuclar)
plt.show()