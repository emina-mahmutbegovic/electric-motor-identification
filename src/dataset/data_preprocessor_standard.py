# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from src.util.shared import num_of_splits

class StandardDataPreprocessor:
    def __init__(self, dataset, row_number, n_splits=num_of_splits):
        self.X = None
        self.y = None
        self.input_cols = []
        self.target_cols = []

        self.dataset = dataset
        self.row_number = row_number
        self.n_splits = n_splits

        # Select every *row_number* row of data
        # Create new columns based on the values of n_k and n_1k columns
        # For each value in the range from 1 to 7, it creates columns with names like n_k_i and n_1k_i,
        # where i is the value.
        self.df = self.dataset.iloc[::self.row_number, :] \
            .assign(**{**{f'n_k_{i}': lambda x: (x.n_k == i).astype(int) for i in range(1, 8)},
                 **{f'n_1k_{i}': lambda x: (x.n_1k == i).astype(int) for i in range(1, 8)}}) \
                .drop(['n_k', 'n_1k'], axis=1)

        # Set currents id_k1 and iq_k1 as targets
        self.target_cols = ['id_k1', 'iq_k1']

        # Set remaining data as inputs
        self.input_cols = [c for c in self.df if c not in self.target_cols]

        # Initialize K-Fold cross validator
        self.cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=2020)

    # Preprocess data for training by using standard scaler
    def preprocess_standard(self):
        # Preprocess the data using standard scaler
        ss_y = StandardScaler().fit(self.df[self.target_cols])
        df = pd.DataFrame(StandardScaler().fit_transform(self.df),
                          columns=self.df.columns)  # actually methodically unsound, but data is large enough

        # Set input and target values
        self.X, self.y = df[self.input_cols].values, df[self.target_cols].values

        return [self.cv, ss_y, self.X, self.y]
    
    
    def pearson(self):
        # Check if data is existing
        if(self.X is None or self.y is None or not self.input_cols or not self.target_cols):
            print("ERROR: Data not preprocessed. Cannot calculate Pearson's coefficient.")
            return
        
        # Define duplicates for filtering out
        duplicates = ['n_k_2', 'n_k_2', 'n_k_3', 'n_k_4', 'n_k_5', 'n_k_6', 'n_k_7', 
                      'n_1k_2', 'n_1k_2', 'n_1k_3', 'n_1k_4', 'n_1k_5', 'n_1k_6', 'n_1k_7']

        results_dict = {}

        for i in range(self.X.shape[1]):  # For each input feature
            for j in range(self.y.shape[1]):  # For each output feature
                corr, p_val = pearsonr(self.X[:, i], self.y[:, j])

                # Update dictionary
                if self.input_cols[i] not in results_dict and self.input_cols[i] not in duplicates:
                    results_dict[self.input_cols[i]] = {}

                # Update dictionary
                # Leave out duplicates
                if(self.input_cols[i] not in duplicates):
                    results_dict[self.input_cols[i]][self.target_cols[j]] = {"correlation": corr, "p_value": p_val}

        return {
            'target_cols': self.target_cols,
            'results_dict': results_dict
        }
