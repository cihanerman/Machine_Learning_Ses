#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:42:50 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('musteriler.csv')

x = data.iloc[:,3:].values

kmeans = KMeans(n_clusters=4,init='k-means++',random_state=0).fit_predict(x)
plt.scatter(x[kmeans == 0,0],x[kmeans == 0,1],s=100, c='red')
plt.scatter(x[kmeans == 1,0],x[kmeans == 1,1],s=100, c='blue')
plt.scatter(x[kmeans == 2,0],x[kmeans == 2,1],s=100, c='green')
plt.scatter(x[kmeans == 3,0],x[kmeans == 3,1],s=100, c='yellow')
plt.show()

ac = AgglomerativeClustering(n_clusters=4,linkage='ward',affinity='euclidean').fit_predict(x)

plt.scatter(x[ac == 0,0],x[ac == 0,1],s=100, c='red')
plt.scatter(x[ac == 1,0],x[ac == 1,1],s=100, c='blue')
plt.scatter(x[ac == 2,0],x[ac == 2,1],s=100, c='green')
plt.scatter(x[ac == 3,0],x[ac == 3,1],s=100, c='yellow')
plt.show()

dendogram = dendrogram(linkage(x,method='ward'))
plt.show()