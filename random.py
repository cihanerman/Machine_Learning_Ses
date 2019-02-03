#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:31:18 2019

@author: cihanerman
"""
# reinforced learning(Pekiştirmeli öğrenme)
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Ads_CTR_Optimisation.csv')

# Random selection (Rastgele seçim)
N = 10000
d = 10
total = 0
selected =[]

for n in range(N):
    ad = random.randrange(d)
    selected.append(ad)
    x = data.values[n,ad]
    total += x

plt.hist(selected,bins=50)
plt.show()