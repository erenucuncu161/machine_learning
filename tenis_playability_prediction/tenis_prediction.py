# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:51:34 2024

@author: erenucuncu161
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
#kodlama
#veriyukleme
veriler=pd.read_csv('C:/Users/erena/Downloads/odev_tenis.csv')
imputer= SimpleImputer(missing_values=np.nan,strategy='mean')


outlok = veriler.iloc[:,0:1].values
print(outlok)

le=preprocessing.LabelEncoder()
outlok[:,0]=le.fit_transform(veriler.iloc[:,0])
print(outlok)
ohe1=preprocessing.OneHotEncoder()
outlok=ohe1.fit_transform(outlok).toarray()
print(outlok)






#encoding

windy= veriler.iloc[:,3:4].values
print(windy)

le1=preprocessing.LabelEncoder()
windy[:,0]=le1.fit_transform(veriler.iloc[:,3])
print(windy)
ohe=preprocessing.OneHotEncoder()
windy=ohe.fit_transform(windy).toarray()
print(windy)

play= veriler.iloc[:,-1:].values
print(play)

le2=preprocessing.LabelEncoder()
play[:,0]=le2.fit_transform(veriler.iloc[:,4])
print(play)
ohe2=preprocessing.OneHotEncoder()
play=ohe2.fit_transform(play).toarray()
print(play)



sonuc=pd.DataFrame(data=outlok,index=range(14),columns=['overcast','rainy','sunny'])
print(sonuc)

sonuc2=pd.DataFrame(data=windy[:,1],index=range(14),columns=['windy'])
print(sonuc2)

sonuc3=pd.DataFrame(data=play[:,1],index=range(14),columns=['play'])
print(sonuc3)


s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred= (regressor.predict(x_test))



play=s2.iloc[:,4].values
print(play)



veri=s2.iloc[:,:5]


x_train, x_test,y_train,y_test=train_test_split(veri,play,test_size=0.33,random_state=0)


r2=LinearRegression()
r2.fit(x_train,y_train)

y_pred= (r2.predict(x_test))

import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=veri,axis=1)

X_l=veri.iloc[:,[0,1,2,3]]
X_l=np.array(X_l,dtype=float)
model=sm.OLS(play,X_l).fit()
print(model.summary())




