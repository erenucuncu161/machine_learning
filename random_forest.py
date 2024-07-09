# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:39:31 2024

@author: erena
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


veriler=pd.read_csv('C:/Users/erena/Downloads/maaslar.csv')
#data frame dilimleme
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#numpy dizi dönüşümü
X=x.values
Y=y.values
Z=X +0.5
K=X-0.4

rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(X,rf_reg.predict(K),color='yellow')

from sklearn.metrics import r2_score
print("random forest R2 degeri")
print(r2_score(Y,rf_reg.predict(X)))
print(r2_score(Y,rf_reg.predict(Z)))
print(r2_score(Y,rf_reg.predict(K)))






