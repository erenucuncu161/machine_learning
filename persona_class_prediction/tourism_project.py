"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df=pd.read_excel('C:/Users/erena/OneDrive/Masaüstü/Projeler/PYTHON/Kurs_Miuul/miuul_gezinomi.xlsx')
df.head()

cat_cols=[col for col in df.columns if str(df[col].dtypes) in ["category","bool","object"]]

num_but_cat=[col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["float64","int64"] ]

cat_but_car = [col for col in df.columns if 
               df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

cat_cols = cat_cols + num_but_cat


cat_cols = [col for col in cat_cols if col not in cat_but_car]

#print(df[cat_cols].nunique())
num_cols=[col for col in df.columns if col not in cat_cols]        
        
def num_summary(dataframe,numerical_col):
    quantiles=[0,0.07,0.30,0.90,1]
    print(dataframe[numerical_col].describe(quantiles).T)


def grab_col_names(dataframe,cat_th=10,car_th=20):
    
    """

    








    

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframedir.
    cat_th : int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th : int,float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri


    Returns
    cat_cols : list
                kategorik değişken listesi
    num_cols : list
                numerik değişken listesi
    cat_but_car : list
                kategorik görünümlü kardinal değişken listesi
    Notes
    -----
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    
    
    
    
    

    """
    
    
    cat_cols=[col for col in df.columns if str(df[col].dtypes) in ["category","bool","object"]]

    num_but_cat=[col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["float64","int64"] ]

    cat_but_car = [col for col in df.columns if 
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

    cat_cols = cat_cols + num_but_cat


    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #print(df[cat_cols].nunique())
    num_cols=[col for col in df.columns if col not in cat_cols]
    
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    return cat_cols,num_cols,cat_but_car

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################")
   

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()
    
cat_cols,num_cols,cat_but_car=grab_col_names(df)
#SORUN 1
print(df.info())
#SORUN 2
print(df["SaleCityName"].unique())

#SORUN 3
print(df["ConceptName"].unique())

#SORUN 4
print(df["ConceptName"].value_counts())

#SORUN 5
print(df.groupby("SaleCityName").agg({"Price": "sum"}))

#SORUN 6
print(df.groupby("ConceptName").agg({"Price": "sum"}))

#SORUN 7
print(df.groupby("SaleCityName").agg({"Price": "mean"}))

#SORUN 8
print(df.groupby("ConceptName").agg({"Price": "mean"}))

#SORUN 9
print(df.groupby(["ConceptName","SaleCityName"]).agg({"Price": "mean"}))





#GÖREV 2
df["SaleCheckInDayDiff_cat"]=pd.cut(df["SaleCheckInDayDiff"],bins=[-1,7,30,90,df["SaleCheckInDayDiff"].max()]
                                ,labels=['Last Minuters','Potential Planners','Planners' ,'Early Bookers'])

#GÖREV 3.1
print(df.groupby(["SaleCityName","ConceptName","SaleCheckInDayDiff_cat"])["Price"].agg(["mean","count"]))


#GÖREV 3.2
print(df.groupby(["SaleCityName","ConceptName","Seasons"])["Price"].agg(["mean","count"]))

#GÖREV 3.3
print(df.groupby(["SaleCityName","ConceptName","CInDay"])["Price"].agg(["mean","count"]))

#GÖREV 4
agg_df = df.groupby(["ConceptName", "SaleCityName","Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False) 
                                    
agg_df.head(20)
print(agg_df)

#GÖREV 5
agg_df.reset_index(inplace=True)

#GÖREV 6
agg_df['sales_level_based'] = agg_df['SaleCityName'].str.upper() + "_" + agg_df['ConceptName'].str.upper() + "_" + agg_df['Seasons'].str.upper()

#GÖREV 7
agg_df["SEGMENT"]=pd.qcut(agg_df["Price"],4,labels=['D','C','B','A']) 
agg_df.head(30)                      
agg_df.groupby("SEGMENT").agg({"Price":["mean","max","count"]})
    
#GÖREV 8
agg_df.sort_values(by="Price")
new_user="ANTALYA_ODA DAHIL_HIGH"
agg_df[agg_df['sales_level_based']== new_user]   





