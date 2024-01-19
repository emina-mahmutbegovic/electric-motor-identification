# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from enum import Enum, unique


# Create options for training linear regression model
@unique
class LinearRegressionOptions(Enum):
    BASIC_TRAINING = 1
    HYPERPARAMETER_TUNING = 2
