#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:48:35 2019

@author: cihanerman
"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from apyori import apriori # __author__ = 'Yu Mochizuki', __author_email__ = 'ymoch.dev@gmail.com'
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('sepet.csv',header = None)

t = []
for i in range(0,7501):
    t.append([str(data.values[i,j]) for j in range(20) if str(data.values[i,j]) != 'nan'])

responce = apriori(t,min_support = 0.01, min_confidence = 0.2, min_lift = 2)

for i in list(responce):
    print(i)