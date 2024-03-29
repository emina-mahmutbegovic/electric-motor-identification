# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from typing import Any

from src.dataset.data_preprocessor_standard import StandardDataPreprocessor


class LinearRegressionModel:
    # Define constructor
    # Inputs:
    # data - preprocessed data used for the training and evaluation
    # row_number - each *row_number* within the dataset will be selected for further processing
    def __init__(self, data, n_splits=5):
        #self.dataset = dataset
        self.n_splits = n_splits

        # Unpack preprocessed data
        self.cv, self.ss_y, self.X, self.y = data

    # Train model and calculate scores
    def train_and_predict(self):
        # Define list for storing the scores
        scores = []

        # Measure training time
        start_time = time.time()

        # Train model and calculate scores
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            ols = LinearRegression().fit(self.X[train_idx], self.y[train_idx])
            pred = ols.predict(self.X[test_idx])
            pred = self.ss_y.inverse_transform(pred)
            gtruth = self.ss_y.inverse_transform(self.y[test_idx])
            scores.append(mean_squared_error(pred, gtruth))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time}")

        return scores

    # Train and predict by using hyperparameter tuning
    # Input
    # cv - cross-validation input strategy
    def hyper_parameter_tuning(self):
        # Define list for storing the scores
        scores = []

        # Step 2: Define the model
        model = LinearRegression()

        # Step 3: Define the hyperparameter grid
        param_grid = {'positive': [True, False]}

        # Step 4: Perform grid search
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=self.n_splits)

        # Measure training time
        start_time = time.time()

        # Train model and caluculate scores
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            grid_search.fit(self.X[train_idx], self.y[train_idx])

            # Step 5: Fit the model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Step 6: Evaluate the results
            y_pred = best_model.predict(self.X[test_idx])
            mse = mean_squared_error(self.y[test_idx], y_pred)

            scores.append(mse)

            print("Best Hyperparameters:", best_params)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time}")

        return scores

    # Print scores
    def format_scores_output(self, scores):
        scores = np.asarray(scores)

        return f'Scores Mean: {scores.mean():.4f} A² +- {2 * scores.std():.4f} A²\nScores Min: {scores.min():.4f} A²\nScores Max: {scores.max():.4f} A²'
