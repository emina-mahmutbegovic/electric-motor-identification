# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, ConvLSTM1D, InputLayer
from src.dataset.data_preprocessor_lstm import LSTMDataPreprocessor
from src.dataset.delayed_input import DelayedInput
from src.util.shared import stop_neural_network_training_flag
import matplotlib.pyplot as plt

class LSTMNetwork:
    def __init__(self, n_units, activation, loss_and_optimizer, metrics, dataset, regressor, num_of_filters, kernel_size, conv_activation):
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

        # Enrich inputs with delayed data
        self.delayed_input = DelayedInput(regressor, self.X, self.y)

        # Split data to train and test
        self.X_train, self.y_train, self.X_test, self.y_test = data_preprocessor.split_data(
            self.delayed_input.inputs, self.delayed_input.targets
        )

        # Reshape data for feeding into LSTM network
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], 1, self.y_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], 1, self.y_test.shape[1]))

        # Extract input shape
        self.num_inputs = np.shape(self.delayed_input.inputs)[1]
        # Extract output shape
        self.num_outputs = np.shape(self.delayed_input.targets)[1]

        # Build model
        self.build_model()

    # Build model
    def build_model(self):
        # Define the LSTM model
        self.model = Sequential()
        #self.model.add(InputLayer(input_shape=(1,1, self.num_inputs)))
        #self.model.add(ConvLSTM1D(filters=self.num_of_filters, kernel_size=self.kernel_size, padding='SAME', return_sequences=True, activation=self.conv_activation,
               #                   go_backwards=True)) 
        #self.model.add(Activation('sigmoid'))
        # self.model.add(ConvLSTM1D(filters=64, kernel_size=4, padding='SAME', return_sequences=True, activation='elu',
        #                           go_backwards=True))
        # self.model.add(ConvLSTM1D(filters=64, kernel_size=4, padding='SAME', return_sequences=True, activation='elu',
        #                           go_backwards=True))
        # self.model.add(ConvLSTM1D(filters=64, kernel_size=4, padding='SAME', return_sequences=True, activation='elu',
        #                           go_backwards=True))
        #self.model.add(Flatten())
        #self.model.add(Reshape((1, self.num_inputs)))  # Reshapes the input to (time_steps, features)

        #self.model.add(LSTM(units=50))

        self.model.add(LSTM(units=int(self.n_units), input_shape=(1, self.num_inputs), return_sequences=True))

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

               # Save model and variables
        model_name = f"{self.optimizer}_lstm.h5"
        self.model.save(model_name)  # Creates a HDF5 file 
        np.save('X.npy', self.X_test)  # Saves the input variables
        np.save('y.npy', self.y_test)  # Saves the target variables
        # Save the scaler to a file
        joblib.dump(self.scaler, 'scaler_lstm.save')


        return history_reports  