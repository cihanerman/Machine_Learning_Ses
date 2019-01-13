# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

data2 = pd.read_csv('eksikveriler.csv')
# print(data2)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
age = data2.iloc[:,1:4].values
# print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)