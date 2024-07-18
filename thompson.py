# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:55:00 2024

@author: erena
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler=pd.read_csv('Ads_CTR_Optimisation.csv')
import random

N=10000
d=10

toplam =0
secilenler=[] 
birler=[0]*d
sifirlar=[0]*d


for n in range(0,N):
    ad=0
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1, sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
       
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1 
    toplam=toplam+odul
    
    
print(toplam)
plt.hist(secilenler)
plt.show()