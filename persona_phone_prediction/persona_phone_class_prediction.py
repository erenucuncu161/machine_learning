import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df=pd.read_csv('C:/Users/erena/OneDrive/Masaüstü/Projeler/PYTHON/Kurs_Miuul/persona.csv')
df.head()
#GÖREV 1
#1.1
df.info()
#1.2
df["SOURCE"].unique()
#1.3
df["PRICE"].nunique()
#1.4
df["PRICE"].value_counts()
#1.5
print(df.groupby("COUNTRY").agg({"PRICE":"count"}))
#1.6
print(df.groupby("COUNTRY").agg({"PRICE":"sum"}))
#1.7
print(df.groupby("SOURCE").agg({"PRICE":"count"}))
#1.8
print(df.groupby("COUNTRY").agg({"PRICE":"mean"}))
#1.9
print(df.groupby("SOURCE").agg({"PRICE":"mean"}))
#1.10
print(df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE":"mean"}))


#GÖREV 2+3
agg_df=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False)
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE",ascending=False))
#GÖREV 4
agg_df.reset_index(inplace=True)
#GÖREV 5
agg_df["AGE_CAT"]=pd.cut(df["AGE"], bins=[0,18,23,30,40,70],
                     labels=['0_18','19_23','24_30','31_40','41_70'])
#GÖREV 6
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x).upper(), axis=1)

agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()



#GÖREV 7
agg_df["SEGMENT"]=pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})
#GÖREV 8
new_user="TUR_ANDROID_MALE_24_30"
agg_df[agg_df["customers_level_based"]==new_user]


