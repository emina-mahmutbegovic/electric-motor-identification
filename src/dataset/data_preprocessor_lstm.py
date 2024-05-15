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
        scaled_inputs = np.array([[a1, a2] for a1, a2 in zip(df['id_k1'], df['iq_k'])])
        scaled = scaler.inverse_transform(scaled_inputs)
        id_k_scaled = [sublist[0] for sublist in scaled]
        time = np.linspace(0, 10, 100)  # 100 time points from 0 to 10
        # Create a plot
        plt.figure(figsize=(10, 5))  # Set the figure size

        # Plot both variables
        plt.plot(time, self.dataset['id_k1'][:100], label='Real', color='blue')
        plt.plot(time, id_k_scaled[:100], label='Scaled', color='red')

        # Adding title and labels
        plt.title('The first 100 input variables over time')
        plt.xlabel('Time')
        plt.ylabel('Value')

        # Add a legend to the plot
        plt.legend()

        # Show the plot
        plt.show()

        # Set input currents
        #id_k_scaled = [sublist[0] for sublist in df]
        #iq_k_scaled = [sublist[1] for sublist in df]

        return [scaler, df]
    
    def split_data(self, X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        return [X_train, y_train, X_test, y_test]

    
    def split_scaled_data(self, train_split_num=0.8):
        # Split into train and test sets
        train_size = int(len(self.preprocessed_data) * train_split_num)

        train, test = self.preprocessed_data[0:train_size,:], self.preprocessed_data[train_size:len(self.preprocessed_data),:]

        return train, test
    
    # Split data into train and test sets
    # Used for not scaled data
    # def split_data(self, train_split_num):
    #     # Set input and target values
    #     X, y = self.dataset[self.input_cols].values, self.dataset[self.target_cols].values

    #     # Split the dataset into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_split_num, random_state=42)

    #     return X_train, y_train, X_test, y_test

    # Create dataset for LSTM
    def create_dataset(self, data, look_back=0):
        dataX, dataY = [], []
        for i in range(len(data)-look_back):
            a = data[i:(i+look_back), 0:self.input_num] 
            dataX.append(a)
            dataY.append(data[i + look_back, -(self.target_num):])
        return np.array(dataX), np.array(dataY)
    
    # Used for not scaled data
    def look_back_data(self, data):
        outputData = []
        for i in range(len(data)-self.look_back):
            a = data[i:(i+self.look_back)] 
            outputData.append(a)
        return np.array(outputData)

    # Preprocess
    def preprocess(self, train_split_num=0.67):
        # Split data
        train, test = self.split_scaled_data(train_split_num)

        # Create train and test data with one step back
        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # Reshape data for feeding into LSTM network
        trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, trainX.shape[2]))
        testX = np.reshape(testX, (testX.shape[0], self.look_back, testX.shape[2]))

        return trainX, trainY, testX, testY
    
    def split_data_to_tensors(self, train_split_num=0.8):
          # Split data
        train, test = self.split_scaled_data(train_split_num)

        # Create train and test data with one step back
        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # Reshape data for feeding into LSTM network
        trainX = np.reshape(trainX, (trainX.shape[0], 0, trainX.shape[2]))
        testX = np.reshape(testX, (testX.shape[0], 0, testX.shape[2]))

        return trainX, trainY, testX, testY


