#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:28:43 2019

@author: cihanerman
"""

# reinforced learning
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import warnings
warnings.filterwarnings('ignore')

# data loading
data = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
 
total = 0
selected = []
ones = [0] * d
zeros = [0] * d

for n in range(1,N):
    ad = 0
    max_th = 0
    for i in range(d):
        ras_beta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if max_th < ras_beta:
            max_th = ras_beta
            ad = i
    selected.append(ad)
    tropy = data.values[n,ad]
    
    if tropy == 1:
        ones[ad] += 1
    else:
        zeros[ad] += 1
        
    total += tropy

print(total)

plt.hist(selected)
plt.show()