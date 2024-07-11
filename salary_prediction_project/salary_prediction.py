# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:22:04 2024

@author: erena
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler=pd.read_csv('C:/Users/erena/Downloads/maaslar_yeni.csv')
veri=veriler.iloc[:,2:]
x=veri.iloc[:,:1]
y=veri.iloc[:,3:4]
#numpy dizi dönüşümü
X=x.values
Y=y.values
#multiple linear regression
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
lin=LinearRegression()
lin.fit(X,Y)
y_pred= (regressor.predict(x_test))


import statsmodels.api as sm

model2=sm.OLS(lin.predict(X),X).fit()
print(model2.summary())


from sklearn.metrics import r2_score
print("MLR R2 degeri")
print(r2_score(Y,lin.predict(X)))

#polynomial regression 

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,Y)
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(x_poly),color='red')
plt.show()

#tahminler
print(lin_reg.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg.predict(poly_reg.fit_transform([[9]])))

model2=sm.OLS(lin_reg.predict(poly_reg.fit_transform(X)),X).fit()
print(model2.summary())

#SVR

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)
print(x_olcekli.shape)
print(y_olcekli.shape)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)



print(svr_reg.predict([[1.7]]))
print(svr_reg.predict([[0.6]]))

from sklearn.metrics import r2_score
print("SVR R2 degeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli).fit()
print(model3.summary())



from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X +0.5
K=X-0.4

model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())
print(r_dt.predict([[1.7]]))
print(r_dt.predict([[1.6]]))

from sklearn.metrics import r2_score
print("decision tree R2 degeri")
print(r2_score(Y,r_dt.predict(X)))

# random forest

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

print('Random Forest OLS')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
from sklearn.metrics import r2_score
print("random forest R2 degeri")
print(r2_score(Y,rf_reg.predict(X)))
print(r2_score(Y,rf_reg.predict(Z)))
print(r2_score(Y,rf_reg.predict(K)))



