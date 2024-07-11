# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:22:00 2024

@author: erena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler=pd.read_csv('C:/Users/erena/Downloads/veriler (1).csv')
x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler


sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
#Logistic regression
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

#SVC

from sklearn.svm import SVC
svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

#naive bases
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

#decision tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

#random forest

from sklearn.ensemble import RandomForestClassifier

rnd=RandomForestClassifier(n_estimators=10,criterion='entropy')
rnd.fit(X_train,y_train)

y_pred=rnd.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

y_proba=rnd.predict_proba(X_test)
  #ROC, TPR , FPR değerleri
  
from sklearn import metrics
fpr , tpr , thold= metrics.roc_curve(y_test, y_proba[:,0],pos_label='e')
  
  
  
  



