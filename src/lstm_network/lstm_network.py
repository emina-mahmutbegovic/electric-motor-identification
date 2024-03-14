# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from keras.models import Sequential
from keras.layers import LSTM, Dense

from src.util.shared import stop_neural_network_training_flag

class LSTMNetwork:
    def __init__(self, n_units, activation, loss_and_optimizer, metrics, data_preprocessor):
        self.n_units = n_units
        self.activation_function = activation
        self.loss = loss_and_optimizer.loss
        self.optimizer = loss_and_optimizer.optimizer
        self.metrics = metrics
        self.data_preprocessor = data_preprocessor

        # Get cross validator, standardized data and split data
        self.trainX, self.trainY, self.testX, self.testY = data_preprocessor.preprocess() 

        self.model = self.build_model()

    # Build model
    def build_model(self):
        n_inputs = self.trainX.shape[2]
        n_outputs = self.trainY.shape[1]

        # Define the LSTM model
        model = Sequential([
            # LSTM layer with n units, input shape is (timesteps, features) per sample
            LSTM(units=int(self.n_units), input_shape=(self.data_preprocessor.look_back, n_inputs)),

            # Dense output layer with softmax activation for classification
            Dense(n_outputs, activation=self.activation_function)
        ])

        return model

    # Compile model
    def compile_model(self):
        # Compile the model
        self.model.compile(optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=self.metrics) 

    # Train model
    def train(self, epochs, batch_size):
        # Preprocess data

        # Define list for storing the history reports
        history_reports = []

        if not stop_neural_network_training_flag.stop:
             # Train model
            history_report = self.model.fit(self.trainX, self.trainY, epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(self.testX, self.testY))
            # Print out process for debug
            print(history_report)

            history_reports.append(str(history_report.history))
        else:
            # Reset flag
            stop_neural_network_training_flag.stop = False

        return history_reports  