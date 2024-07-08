# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:19:42 2024

@author: erena
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


veriler=pd.read_csv('C:/Users/erena/Downloads/maaslar.csv')
#data frame dilimleme
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#numpy dizi dönüşümü
X=x.values
Y=y.values

#polynomial regression
#doğrusal olmayan model oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)),color='red')
plt.show()
#4. dereceden polinom
poly_reg2=PolynomialFeatures(degree=4)
x_poly2=poly_reg2.fit_transform(X)
print(x_poly2)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly2,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)),color='red')
plt.show()
#tahminler
print(lin_reg.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg2.fit_transform([[11]])))

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








