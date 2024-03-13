# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

# This module acts as a shared resource between different parts of the application.

class StopTrainingFlag:
    def __init__(self):
        self.stop = False


# Instantiate the flag at module level, so it's shared across imports
stop_training_flag = StopTrainingFlag()

# Number of data splits
num_of_splits = 5

# Data row for filtering the dataset
data_row = 10
