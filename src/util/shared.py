# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

# This module acts as a shared resource between different parts of the application.

class StopTrainingFlag:
    def __init__(self):
        self.stop = False


# Instantiate the flags at module level, so it's shared across imports
stop_neural_network_training_flag = StopTrainingFlag()
stop_lstm_network_training_flag = StopTrainingFlag()

# Number of data splits
num_of_splits = 5

# Data row for filtering the dataset
data_row = 10

# Define activation functions
activation_functions = ["linear", "relu", "sigmoid", "softmax", "tanh"]

# Define optimizers
optimizers = ["adam", "sgd", "rmsprop"]

# Define loss functions
loss_functions = ["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error"]

# Metrics
metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error"]
