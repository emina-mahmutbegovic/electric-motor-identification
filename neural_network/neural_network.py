# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from dataset.data_preprocessor import DataPreprocessor
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error


class NeuralNetworkShape:
    def __init__(self, input_shape, hidden_units, output_units, activation='linear'):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.activation = activation


class NeuralNetwork:
    def __init__(self, neural_network_shape, dataset, preprocess_row_number=100, n_splits=5):
        self.input_shape = neural_network_shape.input_shape
        self.hidden_units = neural_network_shape.hidden_units
        self.output_units = neural_network_shape.output_units
        self.activation = neural_network_shape.activation
        self.model = self.build_model()

        self.cv, self.ss_y, self.X, self.y = DataPreprocessor(dataset, preprocess_row_number, n_splits).preprocess()

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        for units in self.hidden_units:
            model.add(layers.Dense(units, activation=self.activation))

        model.add(layers.Dense(self.output_units, activation='linear'))  # Change activation based on the task
        return model

    def compile_model(self, optimizer='sgd', loss='mse', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, epochs, batch_size, validation_data=None):
        # Define list for storing the history reports
        history_reports = []

        for train_idx, test_idx in self.cv.split(self.X, self.y):
            # Train model
            history_report = self.model.fit(self.X[train_idx], self.y[train_idx], epochs=epochs, batch_size=batch_size,
                                     validation_data=validation_data)
            print(history_report)
            history_reports.append(str(history_report.history))

        return history_reports

    def evaluate(self):
        # Define list for storing the evaluation reports
        eval_reports = []

        for train_idx, test_idx in self.cv.split(self.X, self.y):
            # Evaluate model
            eval_report = self.model.evaluate(self.X[test_idx], self.y[test_idx])

            print(eval_report)
            eval_reports.append(eval_report)

        return eval_reports

    def predict(self):
        return self.model.predict(self.X)
