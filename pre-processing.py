"""
PES1201800395
PES1201801549
PES1201801618
Code to clean the dataset provided
"""

# Importing required libraries
import numpy as np
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

# Pre-processing function
def preprocessing(df):
    # splitting the dataframe into 2 based on outcome of LBW
    df1 = df[df.Result == 1]
    df2 = df[df.Result == 0]

    # using mean to replace missing values
    df1['HB'].replace(np.nan, np.mean(df1['HB']), inplace=True)
    df1['Weight'].replace(np.nan, np.mean(df1['Weight']), inplace=True)
    df1['Age'].replace(np.nan, np.mean(df1['Age']), inplace=True)
    df2['HB'].replace(np.nan, np.mean(df2['HB']), inplace=True)
    df2['Weight'].replace(np.nan, np.mean(df2['Weight']), inplace=True)
    df2['Age'].replace(np.nan, np.mean(df2['Age']), inplace=True)

    # using median to replace missing values
    df1['BP'].replace(np.nan, np.nanmedian(df1['BP']), inplace=True)
    df2['BP'].replace(np.nan, np.nanmedian(df2['BP']), inplace=True)

    # using mode to replace missing values
    df1['Delivery phase'].replace(np.nan, df1['Delivery phase'].mode()[0], inplace=True)
    df1['Education'].replace(np.nan, df1['Education'].mode()[0], inplace=True)
    df1['Residence'].replace(np.nan, df1['Residence'].mode()[0], inplace=True)
    df2['Delivery phase'].replace(np.nan, df2['Delivery phase'].mode()[0], inplace=True)
    df2['Education'].replace(np.nan, df2['Education'].mode()[0], inplace=True)
    df2['Residence'].replace(np.nan, df2['Residence'].mode()[0], inplace=True)

    # combining the two pre-processed dataframes
    combine = [df1, df2]
    newdf = pd.concat(combine)

    return newdf

# Reading the dataset
df = pd.read_csv("LBW_Dataset.csv")

# Calling the preprocessing function
df = preprocessing(df)

# Saving the preprocessed dataframe
df.to_csv("cleanedLBW.csv")
