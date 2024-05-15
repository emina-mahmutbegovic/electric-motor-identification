# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from sklearn.model_selection import KFold
class KFoldValidator:
    def __init__(self, num_of_splits = 5):
        self.num_of_splits = num_of_splits
        # Initialize K-Fold cross validator
        self.cv = KFold(n_splits=num_of_splits, shuffle=True, random_state=2020)