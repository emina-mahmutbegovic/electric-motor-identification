# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
from tensorflow import keras
from keras import layers

from src.dataset.data_preprocessor_standard import StandardDataPreprocessor
from src.util.shared import stop_neural_network_training_flag

from src.dataset.data_preprocessor_standard import StandardDataPreprocessor
from src.dataset.k_fold import KFoldValidator
import joblib

class NeuralNetworkShape:
    def __init__(self, hidden_units, output_units):
        self.hidden_units = hidden_units
        self.output_units = output_units


class NeuralNetworkActivation:
    def __init__(self, hidden_layer_activation='linear', output_activation='linear'):
        self.hidden_layer_activation = hidden_layer_activation
        self.output_activation = output_activation


class NeuralNetworkLossAndOptimizer:
    def __init__(self, loss='mse', optimizer='sgd'):
        self.loss = loss
        self.optimizer = optimizer


class NeuralNetwork:
    def __init__(self, shape, activation, loss_and_optimizer, metrics, data):
        self.hidden_units = shape.hidden_units
        self.output_units = shape.output_units
        self.hidden_layer_activation = activation.hidden_layer_activation
        self.output_activation = activation.output_activation
        self.loss = loss_and_optimizer.loss
        self.optimizer = loss_and_optimizer.optimizer
        self.metrics = metrics

        self.data = data

        # Initialize data preprocessor
        data_preprocessor_standard = StandardDataPreprocessor(self.data.reduced_data)
        # Preprocess data
        self.ss_y, df = data_preprocessor_standard.preprocess_standard()

        # Initialize K-Fold cross validator
        self.cv = KFoldValidator().cv

        # Split preprocessed data into inputs and targets
        self.X, self.y = data.split_data(df)

        # Extract input shape
        self.input_shape = (np.shape(self.X)[1],)

        # Build model
        self.model = self.build_model()

        # Initialize variables for testing the model
        self.X_test = []
        self.y_test = []

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        for units in self.hidden_units:
            model.add(layers.Dense(units, activation=self.hidden_layer_activation))

        model.add(layers.Dense(self.output_units, activation=self.output_activation))
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        print("Broj parametara u modelu:", self.model.count_params())

    def train(self, epochs, batch_size, validation_data=None):
        # Define list for storing the history reports
        history_reports = []

        for train_idx, test_idx in self.cv.split(self.X, self.y):
            if not stop_neural_network_training_flag.stop:
                # Train model
                history_report = self.model.fit(self.X[train_idx], self.y[train_idx], epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=validation_data)
                # Print out process for debug
                print(history_report)

                history_reports.append(str(history_report.history))

                # Save values for testing
                self.X_test.append(self.X[test_idx])
                self.y_test.append(self.y[test_idx])

            else:
                # Reset flag
                stop_neural_network_training_flag.stop = False
                break
        
        # Save model and variables
        model_name = f"{self.optimizer}_fnn.h5"
        self.model.save(model_name)  # Creates a HDF5 file 
        np.save('X_test.npy', self.X_test)  # Saves the input variables for testing
        np.save('y_test.npy', self.y_test)  # Saves the target variables for testing
        # Save the scaler to a file
        joblib.dump(self.ss_y, 'scaler.save')

        return history_reports
