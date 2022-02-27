import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

def fill_nan_with_mean_most(df):
    df = df.copy()
    s = pd.Series( df.isna().sum()>0 )
    have_null_cols  = s[s].index
    #print(have_null_cols)
    for i in range(len(have_null_cols)):
        now_col = df[ have_null_cols[i] ]
        if (now_col.dtypes=="object"):
            #print(now_col.mode().mode().values[0])
            df[ have_null_cols[i] ] = now_col.fillna(now_col.mode().values[0])
        else:
            #print(now_col.mean())
            df[ have_null_cols[i] ] = now_col.fillna(now_col.mean())
            #df[ have_null_cols[i] ] = now_col.fillna(now_col.rolling(50,min_periods=1).sum(std=3).mean())
    return df

def labeling(df):
    df = df.copy()
    
    ## 有序用 label encoding
    ## 無序用ONE HOT(dummies)
    df['euducation_level'] = labelencoder.fit_transform(df['euducation_level'])
    df['previous_connect_month'] = labelencoder.fit_transform(df['previous_connect_month'])
    df['previous_connect_weekday'] = labelencoder.fit_transform(df['previous_connect_weekday'])
    print("---label encode", df.shape)
    df = pd.get_dummies(df)
    print("---one hot encoding", df.shape)
    return df
    
def preprocessing_mean_mode(df):
    df_without_null = fill_nan_with_mean_most(df)
    df_label = labeling(df_without_null)
    
    return df_label

def preprocessing_only_label(df):
    df = df.copy()
    df = df.dropna(axis=0)
    for feature in df.columns:
        df[feature] = labelencoder.fit_transform(df[feature])
    return df

def preprocessing_onehot(df):
    df = df.copy()
    df = df.dropna(axis=0)
    df = pd.get_dummies(df)
    return df