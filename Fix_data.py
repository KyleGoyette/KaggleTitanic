import csv
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os

pd.set_option('display.mpl_style', 'default')
pd.set_option('interactive', 'False')
plt.interactive(False)

train_df = pd.read_csv("./datasets/train.csv", dtype={"Age": np.float64},)
#print train_df.info()
#print train_df.head()

#print train_df.info()

def maybe_fix_data(filepath,test=False):
    fname,ext=os.path.splitext(filepath)

    if True:# os.path.isfile(fname+'_fixed'+ext):
        df=pd.read_csv(filepath)
        df=df.drop(['Name', 'Ticket','Cabin'], axis=1)
        df['Embarked']=df['Embarked'].fillna("S")
        df['Age']=df['Age'].replace("NaN")
        df['Age']=df['Age'].astype(int)
        #dead=df['Survived']==0



        if test==False:
        #set categories to numeric
            df.dropna()
            df=pd.get_dummies(df,columns=['Survived'])
            df['Survived_1']=df['Survived_1'].astype(int)
            df['Survived_0']=df['Survived_0'].astype(int)
            df=df.drop(['PassengerId'],axis=1)

        df=pd.get_dummies(df,columns=['Sex'])
        df=pd.get_dummies(df,columns=['Embarked'])
        df['Sex_male']=df['Sex_male'].astype(int)
        df['Sex_female']=df['Sex_female'].astype(int)
        df['Embarked_Q']=df['Embarked_Q'].astype(int)
        df['Embarked_S']=df['Embarked_S'].astype(int)
        df['Embarked_C']=df['Embarked_C'].astype(int)

        print df

        df1=df[:790]
        df2=df[791:]


        df1.to_csv(fname+'_fixed'+ext)
        df2.to_csv(fname+ '_crossval'+ext)
#train_df=train_df.drop(['PassengerId','Name', 'Ticket'], axis=1)

#train_df["Embarked"]=train_df["Embarked"].fillna("S")

#train_df.to_csv("./datasets/train_fixed.csv")


maybe_fix_data('./datasets/train.csv')
#maybe_fix_data('./datasets/test.csv',test=True)

