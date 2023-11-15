# artorg-phd
This repository contains my solution to the Artog-PHD interview test. 

The repository is not yet complete.It will be updated constantly, with the README.md describing every bits of it. 

The following sub-sections will describe problem, the dataset, proposed solution and the current scripts in the repository.





# Description of the problem:
This challenge is about Blood Glucose prediction. It is a time-series forecasting problem. Blood Glucose Prediction(BGM) valuable information for improving the insulin management for patients with type 1 diabetes (PwT1D). The goal of this challenge is to develop a deep learning model to predict blood glucose level of individuals with type1 diabetes. A key feature of a time series data is the fact that they're ordered sequences/representation of information such that a current timestep depends on the previous one. This makes them for suitable for sequence based models like RNNs and its variants(LSTM etc). This is because RNNs were built to handle time-dependent/sequence-based data like this, where a current time step is dependent on the previous one. However, attention-based networks have been rising in poplularity over the last few years, for their promising performance across my tasks. 

In particular, this time-series problem for this challenge is multivariate. This is because multiple variables are observed/recorded simulateneously at each given time step. 

# About the dataset:

The dataset used is the OhioT1DM dataset for blood glucose prediction. It is a publicly available dataset containing continuous glucose monitoring (CGM) data, insulin doses, self-reported life-event data, and physiological sensor data for 12 people with type 1 diabetes. The dataset records information at a 5mintues interval for a period of eight weeks for each individual. This means about 12 times per hour.  It includes information such as blood glucose level(cgm), basal rate, galvanic skin rate, hear rate insulin doses, meal times, carbohydrate estimates, exercise, sleep, work, stress, etc. 

More information and to downloading the dataset can be don via the official dataset webpage here: [OhioT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).


# About the Scripts

For this challenge, the dataset was provided and downloaded in a zip format.

To load the dataset into a single csv for train and test, run: 

```
python load_data.py
```
This script assumes the data has already been downloaded in .zip format. The script extracts the subfolders and transforms the data into single .csv files for both train and test

The ```preprocess.py``` script preprocesses the data, impute missing values, perform transformstions for modelling. It can be directly accessed from ```train.py``` without a separate call.
This script also includes transforming the data to represent a 1hour timeframe. The reason for this is to reduce noise and avoid drastic change for the model.

The current ```train.py``` script only implements a simple LSTM as a baseline model to compare with other state-of-the-art such as [GluPred](https://ieeexplore.ieee.org/document/9474665) and Autoformer described below.


# Description of the Proposed Solution

For this challenge, the proposed solution is the [Autoformer](https://huggingface.co/docs/transformers/model_doc/autoformer) transformer model for time series. 
The [paper](https://arxiv.org/abs/2205.13504) was recently presented at AAAI 2023 and a previous version had won the AAAI 2021 paper award. 

The model achieved state-of-the-art results on benchmarks when compared to previous method. Notably, one of the benchmarks is very closly related to the task for this challenge Weekly reported disease for Influenza ([ILI](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html))
