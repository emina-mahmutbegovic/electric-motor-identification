# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMDataPreprocessor: 
    def __init__(self, dataset):
        self.dataset = dataset

        # Set currents id_k1 and iq_k1 as targets
        self.target_cols = ['id_k1', 'iq_k1']

        # Target num
        self.target_num = len(self.target_cols)

        # Set remaining data as inputs
        self.input_cols = [c for c in dataset if c not in self.target_cols]
        # Input num
        self.input_num = len(self.input_cols)
    
    def preprocess_min_max(self): 
        # Scale data by using MinMax scaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(self.dataset[self.target_cols])

        df = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(self.dataset), columns=self.dataset.columns)

        return [scaler, df]
    
    def split_data(self, X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        return [X_train, y_train, X_test, y_test]


