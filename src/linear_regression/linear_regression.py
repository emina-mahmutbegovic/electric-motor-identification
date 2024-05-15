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
from src.dataset.k_fold import KFoldValidator
from src.dataset.delayed_input import DelayedInput
import matplotlib.pyplot as plt

class LinearRegressionModel:
    # Define constructor
    # Inputs:
    # data - dataset
    def __init__(self, data, regressor):
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

        # Variables for storing models
        self.model = None
        self.best_model = None

    # Train model and calculate scores
    def train(self):
        # Define list for storing the scores
        scores = []

        # Measure training time
        start_time = time.time()

        # Train model and calculate scores
        for train_idx, test_idx in self.cv.split(self.delayed_input.inputs, self.delayed_input.targets):
            self.model = LinearRegression().fit(self.delayed_input.inputs[train_idx], self.delayed_input.targets[train_idx])
            pred = self.model.predict(self.delayed_input.inputs[test_idx])
            pred = self.ss_y.inverse_transform(pred)
            gtruth = self.ss_y.inverse_transform(self.delayed_input.targets[test_idx])
            mse = mean_squared_error(pred, gtruth)
            rmse = np.sqrt(mse)
            scores.append(rmse)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time}")

        # plt.plot(scores, label='Train Loss')
        # plt.title('Training Curve')
        # plt.ylabel('RMSE')
        # plt.xlabel('Epoch')
        # plt.legend(loc='upper right')
        # plt.show()

        return scores
    
    def predict(self):
        # Predict results
        pred = self.model.predict(self.delayed_input.inputs)

        # Perform inverse transformation
        pred = self.ss_y.inverse_transform(pred)
        gtruth = self.ss_y.inverse_transform(self.delayed_input.targets)

        id_k1 = [sublist[0] for sublist in gtruth[-400:]]
        iq_k1 = [sublist[1] for sublist in gtruth[-400:]]

        id_k_pred = [sublist[0] for sublist in pred[-400:]]
        iq_k_pred = [sublist[1] for sublist in pred[-400:]]

        # Plot results
        self.data.plot_results(id_k1, id_k_pred, 'Variable id_k+1', 'Predicted variable id_k+1')
        self.data.plot_results(iq_k1, iq_k_pred, 'Variable iq_k+1', 'Predicted variable iq_k+1')


    # Train and predict by using hyperparameter tuning
    # Input
    # cv - cross-validation input strategy
    def hyper_parameter_tuning_train(self):
        # Define list for storing the scores
        scores = []

        # Step 2: Define the model
        model = LinearRegression()

        # Step 3: Define the hyperparameter grid
        param_grid = {}

        # Step 4: Perform grid search
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)

        # Measure training time
        start_time = time.time()

        # Train model and caluculate scores
        for train_idx, test_idx in self.cv.split(self.delayed_input.inputs, self.delayed_input.targets):
            grid_search.fit(self.delayed_input.inputs[train_idx], self.delayed_input.targets[train_idx])

            # Step 5: Fit the model
            self.best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Step 6: Evaluate the results
            y_pred = self.best_model.predict(self.delayed_input.inputs[test_idx])
            y_pred = self.ss_y.inverse_transform(y_pred)
            gtruth = self.ss_y.inverse_transform(self.delayed_input.targets[test_idx])
            mse = mean_squared_error(y_pred, gtruth)
            rmse = np.sqrt(mse)

            scores.append(rmse)

            print("Best Hyperparameters:", best_params)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time}")

        return scores
    
    def hyperparameter_tuning_predict(self):
        # Predict results
        pred = self.best_model.predict(self.delayed_input.inputs)
        pred = self.ss_y.inverse_transform(pred)
        gtruth = self.ss_y.inverse_transform(self.delayed_input.targets)

        id_k1 = [sublist[0] for sublist in gtruth]
        iq_k1 = [sublist[1] for sublist in gtruth]

        id_k_pred = [sublist[0] for sublist in pred]
        iq_k_pred = [sublist[1] for sublist in pred]

        # Plot results
        self.data.plot_results(id_k1, id_k_pred, 'Variable id_k+1', 'Predicted variable id_k+1')
        self.data.plot_results(iq_k1, iq_k_pred, 'Variable iq_k+1', 'Predicted variable iq_k+1')

    # Print scores
    def format_scores_output(self, scores):
        scores = np.asarray(scores)

        return f'Scores Mean: {scores.mean():.4f} A² +- {2 * scores.std():.4f} A²\nScores Min: {scores.min():.4f} A²\nScores Max: {scores.max():.4f} A²'
