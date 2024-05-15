# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import joblib
from keras.layers import Dense, ConvLSTM1D, InputLayer
from src.dataset.data_preprocessor_lstm import LSTMDataPreprocessor
from src.util.shared import stop_neural_network_training_flag

class ConvLSTMNetwork:
    def __init__(self, n_units, activation, loss_and_optimizer, metrics, dataset, num_of_filters, kernel_size, conv_activation):
        self.n_units = n_units
        self.activation_function = activation
        self.loss = loss_and_optimizer.loss
        self.optimizer = loss_and_optimizer.optimizer
        self.metrics = metrics
        self.data = dataset
        self.num_of_filters = num_of_filters
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.model = None

        # Initialize data preprocessor
        data_preprocessor = LSTMDataPreprocessor(self.data.reduced_data)
        # Preprocess data
        self.scaler, df = data_preprocessor.preprocess_min_max()

        # Split preprocessed data into inputs and targets
        self.X, self.y = self.data.split_data(df)

        # Split data to train and test
        self.X_train, self.y_train, self.X_test, self.y_test = data_preprocessor.split_data(
            self.X, self.y
        )

        # Reshape data for feeding into ConvLSTM network
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, 1, self.X_train.shape[1]))
        self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], 1, 1, self.y_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, 1, self.X_test.shape[1]))
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], 1, 1, self.y_test.shape[1]))

        # Extract input shape
        self.num_inputs = np.shape(self.X)[1]
        # Extract output shape
        self.num_outputs = np.shape(self.y)[1]

        # Build model
        self.build_model()

    # Build model
    def build_model(self):
        # Define the ConvLSTM model
        self.model.add(InputLayer(input_shape=(1,1, self.num_inputs)))
        self.model.add(ConvLSTM1D(filters=self.num_of_filters, kernel_size=self.kernel_size, padding='SAME', return_sequences=True, activation=self.conv_activation,
                                  go_backwards=True)) 

        self.model.add(Dense(self.num_outputs, activation=self.activation_function))
        print(self.model.summary())


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
            history_report = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                batch_size=batch_size,
                                validation_data=None)
            # Print out process for debug
            print(history_report)

            history_reports.append(str(history_report.history))
        else:
            # Reset flag
            stop_neural_network_training_flag.stop = False

               # Save model and variables
        model_name = f"{self.optimizer}_conv_lstm.h5"
        self.model.save(model_name)  # Creates a HDF5 file 
        np.save('X.npy', self.X_test)  # Saves the input variables for testing
        np.save('y.npy', self.y_test)  # Saves the target variables for testing
        # Save the scaler to a file
        joblib.dump(self.scaler, 'scaler_conv_lstm.save')


        return history_reports  