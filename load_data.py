#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
import glob



#Unzip downloaded FIle

!unzip Ohio_Data.zip



# Extract training data
train_data = []
for year in ['2018', '2020']:
    for filepath in glob.glob(f'Ohio{year}_processed/train/*.csv'):
        basepath, file = os.path.split(filepath)
        filename = os.path.splitext(file)[0]
        #print(filename)
        patient_id = filename.split('-')[0]
        data = pd.read_csv(filepath)

        data['patient_id'] = patient_id
        data['year'] = year
        train_data.append(data)

train_data = pd.concat(train_data, ignore_index=True)
train_data.to_csv('train.csv', index=False)

# Extract testing data
test_data = []
for year in ['2018', '2020']:
    for filepath in glob.glob(f'Ohio{year}_processed/test/*.csv'):
        basepath, file = os.path.split(filepath)
        filename = os.path.splitext(file)[0]
        patient_id = filename.split('-')[0]
        
        data = pd.read_csv(filepath)
        data['patient_id'] = patient_id
        data['year'] = year
        test_data.append(data)

test_data = pd.concat(test_data, ignore_index=True)
test_data.to_csv('test.csv', index=False)
