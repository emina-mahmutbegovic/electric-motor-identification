# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from typing import Any

from src.dataset.data_preprocessor import DataPreprocessor


class LinearRegressionModel:
    # Define constructor
    # Inputs:
    # dataset - dataset for the training
    # row_number - each *row_number* within the dataset will be selected for further processing
    def __init__(self, dataset, preprocess_row_number=100, n_splits=5):
        self.dataset = dataset
        self.n_splits = n_splits

        self.cv, self.ss_y, self.X, self.y = DataPreprocessor(dataset, preprocess_row_number, n_splits).preprocess()

    # Train model and calculate scores
    def train_and_predict(self):
        # Define list for storing the scores
        scores = []

        # Train model and caluculate scores
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            ols = LinearRegression().fit(self.X[train_idx], self.y[train_idx])
            pred = ols.predict(self.X[test_idx])
            pred = self.ss_y.inverse_transform(pred)
            gtruth = self.ss_y.inverse_transform(self.y[test_idx])
            scores.append(mean_squared_error(pred, gtruth))

        return scores

    # Train and predict by using hyperparameter tuning
    # Input
    # cv - cross-validation input strategy
    def hyper_parameter_tuning(self):
        # Define list for storing the scores
        scores = []

        # Define variable to store best model
        bestModel: Any

        # Step 2: Define the model
        model = LinearRegression()

        # Step 3: Define the hyperparameter grid
        param_grid = {'positive': [True, False]}

        # Step 4: Perform grid search
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=self.n_splits)

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

        return scores

    # Print scores
    def format_scores_output(self, scores):
        scores = np.asarray(scores)

        return f'Scores Mean: {scores.mean():.4f} A² +- {2 * scores.std():.4f} A²\nScores Min: {scores.min():.4f} A²\nScores Max: {scores.max():.4f} A²'
