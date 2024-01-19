# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from tensorflow import keras
from tensorflow.keras import layers

from src.dataset.data_preprocessor import DataPreprocessor


class NeuralNetworkShape:
    def __init__(self, input_shape, hidden_units, output_units):
        self.input_shape = input_shape
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
    def __init__(self, shape, activation, loss_and_optimizer, dataset, preprocess_row_number=100, n_splits=5):
        self.input_shape = shape.input_shape
        self.hidden_units = shape.hidden_units
        self.output_units = shape.output_units
        self.hidden_layer_activation = activation.hidden_layer_activation
        self.output_activation = activation.output_activation
        self.loss = loss_and_optimizer.loss
        self.optimizer = loss_and_optimizer.optimizer
        self.model = self.build_model()

        # Get cross validator, standardize data and split data
        self.cv, self.ss_y, self.X, self.y = DataPreprocessor(dataset, preprocess_row_number, n_splits).preprocess()

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        for units in self.hidden_units:
            model.add(layers.Dense(units, activation=self.hidden_layer_activation))

        model.add(layers.Dense(self.output_units, activation=self.output_activation))
        return model

    def compile_model(self, metrics=['accuracy']):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)

    def train(self, label, epochs, batch_size, validation_data=None):
        # Define list for storing the history reports
        history_reports = []

        for train_idx, test_idx in self.cv.split(self.X, self.y):
            # Train model
            history_report = self.model.fit(self.X[train_idx], self.y[train_idx], epochs=epochs, batch_size=batch_size,
                                     validation_data=validation_data)
            # Print out process for debug
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
