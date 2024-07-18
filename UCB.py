# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:33:30 2024

@author: erena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler=pd.read_csv('Ads_CTR_Optimisation.csv')
import random
#random selection
'''
N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    toplam=toplam+odul
    
plt.hist(secilenler)
plt.show()
'''
import math
#UCB
N=10000
d=10
oduller=[0]*d
tiklamalar=[0]*d
toplam =0
secilenler=[] 


for n in range(0,N):
    ad=0
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if(max_ucb<ucb):
            max_ucb=ucb
            ad=i
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
    
print(toplam)
plt.hist(secilenler)
plt.show()

    
    
    