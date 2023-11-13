#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(path):

    """This function preprocesses the dataset and imputes missing values
    for cbg: missing values are forward filled
    for other columns: missing values are replace with 0 because they don't exit
      """

    df = pd.read_csv(path)
    #Convert time stamp col
    df['5minute_intervals_timestamp'] = pd.datetime(df['5minute_intervals_timestamp'], unit='m')
    

    #set index to timestampt
    df.set_index('5minute_intervals_timestamp', inplace=True)
    #Resample to a timeframe of 1h
    df = df.resample("1H").mean()
    # Drop the 'missing_cgm' column
    df = df.drop(columns=['missing_cbg','year','patient_id'])

    #check for columns with missing values
    missing_values = df.isnull().any()
    missing_cols = missing_values[missing_values].index.tolist()

    for col in missing_cols:
        #print(col)
        if col == 'cbg':
            # Impute any missing values in the 'cgm' column
            df[col] = df[col].fillna(method='ffill')
        else:
            #impute other columns with 0
            df[col] = df[col].fillna(0)


    return df



def transform_data(path):

    """This method scales the feature and target variables using Min-Max scaling"""
    df = preprocess_data(path)
    scaler = MinMaxScaler()  

    # Min-Max scaling
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled
