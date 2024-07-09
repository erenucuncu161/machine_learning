# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:19:42 2024

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


from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

print(svr_reg.predict([[1.7]]))
print(svr_reg.predict([[-0.6]]))

from sklearn.metrics import r2_score
print("SVR R2 degeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))






