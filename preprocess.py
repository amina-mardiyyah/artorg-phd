import pandas as pd
import numpy as np
import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader



class Preprocess:
  def __init__(self,path):
    self.df = pd.read_csv(path)
    

  def preprocess_data(self):
    
    """This function preprocesses the dataset and imputes missing values
    for cbg: missing values are forward filled
    for other columns: missing values are replace with 0 because they don't exit
      """
    df = self.df
    
    #Convert time stamp col
    df['5minute_intervals_timestamp'] = df['5minute_intervals_timestamp'].astype(int)
    
    #set index to timestampt
    df = df.set_index('5minute_intervals_timestamp')
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




def transform_data(self):
        """This method scales the feature and target variables using Min-Max scaling"""
        self.df = preprocess_data(self.df)
        scaler = MinMaxScaler()  
        
        # Min-Max scaling
        df_scaled = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        
        return df_scaled
    
