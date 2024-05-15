# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Any

from src.dataset.data_preprocessor_standard import StandardDataPreprocessor
from src.dataset.k_fold import KFoldValidator

class LinearRegressionModel:
    # Define constructor
    # Inputs:
    # data - dataset
    def __init__(self, data):
        self.data = data

        # Initialize data preprocessor
        data_preprocessor_standard = StandardDataPreprocessor(self.data.reduced_data)
        # Preprocess data
        self.ss_y, df = data_preprocessor_standard.preprocess_standard()

        # Initialize K-Fold cross validator
        self.cv = KFoldValidator().cv

        # Split preprocessed data into inputs and targets
        self.X, self.y = data.split_data(df)

        # Variable for storing models
        self.model = None

    # Train model and calculate scores
    def train(self):
        # Define list for storing the scores
        scores = []

        # Measure training time
        start_time = time.time()

        # Train model and calculate scores
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            self.model = LinearRegression().fit(self.X[train_idx], self.y[train_idx])
            pred = self.model.predict(self.X[test_idx])
            pred = self.ss_y.inverse_transform(pred)
            gtruth = self.ss_y.inverse_transform(self.y[test_idx])
            mse = mean_squared_error(pred, gtruth)
            rmse = np.sqrt(mse)
            scores.append(rmse)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time}")

        return scores

    # Print scores
    def format_scores_output(self, scores):
        scores = np.asarray(scores)

        return f'Scores Mean: {scores.mean():.4f} A² +- {2 * scores.std():.4f} A²\nScores Min: {scores.min():.4f} A²\nScores Max: {scores.max():.4f} A²'
