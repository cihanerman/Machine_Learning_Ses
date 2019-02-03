#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:59:59 2019

@author: cihanerman
"""
# reinforced learning(Pekiştirmeli öğrenme)
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
tropys = [0] * d
clicks = [0] * d
total = 0
selected = []

for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(d):
        if clicks[i] > 0:
            mean = tropys[i] / clicks[i]
            delta = math.sqrt(3 / 2 * math.log(n) / clicks[i])
            ucb = mean + delta
        else:
            ucb = N * 10
            
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    selected.append(ad)
    clicks[ad] += 1
    tropy = data.values[n,ad]
    tropys[ad] += tropy
    total += tropy

print(total)

plt.hist(selected)
plt.show()