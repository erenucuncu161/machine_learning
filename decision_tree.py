# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:46:37 2024

@author: erena
"""


import pandas as pd
import matplotlib.pyplot as plt



veriler=pd.read_csv('C:/Users/erena/Downloads/maaslar.csv')
#data frame dilimleme
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#numpy dizi dönüşümü
X=x.values
Y=y.values
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X +0.5
K=X-0.4

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')
plt.plot(X,r_dt.predict(Z),color='yellow')
plt.plot(X,r_dt.predict(K),color='green')
print(r_dt.predict([[1.7]]))
print(r_dt.predict([[1.6]]))

from sklearn.metrics import r2_score
print("decision tree R2 degeri")
print(r2_score(Y,r_dt.predict(X)))





