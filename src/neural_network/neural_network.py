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
from src.dataset.delayed_input import DelayedInput
import joblib
import matplotlib.pyplot as plt

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
    def __init__(self, shape, activation, loss_and_optimizer, metrics, data, regressor):
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

        # Enrich inputs with delayed data
        self.delayed_input = DelayedInput(regressor, self.X, self.y)

        # Extract input shape
        self.input_shape = (np.shape(self.delayed_input.inputs)[1],)

        # Build model
        self.model = self.build_model()

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

        for train_idx, test_idx in self.cv.split(self.delayed_input.inputs, self.delayed_input.targets):
            if not stop_neural_network_training_flag.stop:
                # Train model
                history_report = self.model.fit(self.delayed_input.inputs[train_idx], self.delayed_input.targets[train_idx], epochs=epochs,
                                                batch_size=batch_size,
                                                validation_data=validation_data)
                # Print out process for debug
                print(history_report)

                
                # loss_rmse = np.sqrt(history_report.history['loss'])
                # plt.plot(loss_rmse, label='Train Loss')
                # plt.title('Training Curve')
                # plt.ylabel('RMSE')
                # plt.xlabel('Epoch')
                # plt.legend(loc='upper right')
                # plt.show()

                history_reports.append(str(history_report.history))
            else:
                # Reset flag
                stop_neural_network_training_flag.stop = False
                break
        
        # Save model and variables
        model_name = f"{self.optimizer}_fnn.h5"
        self.model.save(model_name)  # Creates a HDF5 file 
        np.save('X.npy', self.delayed_input.inputs)  # Saves the input variables
        np.save('y.npy', self.delayed_input.targets)  # Saves the target variables
        # Save the scaler to a file
        joblib.dump(self.ss_y, 'scaler.save')

        return history_reports

    def evaluate(self):
        # Define list for storing the evaluation reports
        eval_reports = []

        for train_idx, test_idx in self.cv.split(self.delayed_input.inputs, self.delayed_input.targets):
            # Evaluate model
            eval_report = self.model.evaluate(self.delayed_input.inputs[test_idx], self.delayed_input.targets[test_idx])

            print(eval_report)
            eval_reports.append(eval_report)

        return eval_reports

    def predict(self):
        # Predict results
        pred = self.model.predict(self.delayed_input.inputs)

        # Perform inverse transformation
        pred = self.ss_y.inverse_transform(pred)
        gtruth = self.ss_y.inverse_transform(self.delayed_input.targets)

        id_k1 = [sublist[0] for sublist in gtruth]
        iq_k1 = [sublist[1] for sublist in gtruth]

        id_k_pred = [sublist[0] for sublist in pred]
        iq_k_pred = [sublist[1] for sublist in pred]

        # Plot results
        self.data.plot_results(id_k1, id_k_pred, 'Variable id_k+1', 'Predicted variable id_k+1')
        self.data.plot_results(iq_k1, iq_k_pred, 'Variable iq_k+1', 'Predicted variable iq_k+1')
