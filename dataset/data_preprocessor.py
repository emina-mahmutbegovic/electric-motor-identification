# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, dataset, row_number, n_splits=5):
        self.dataset = dataset
        self.row_number = row_number
        self.n_splits = n_splits

    # Preprocess data for training
    # Select every *row_number* row of data
    # Create new columns based on the values of n_k and n_1k columns
    # For each value in the range from 1 to 7, it creates columns with names like n_k_i and n_1k_i,
    # where i is the value.
    def preprocess(self):
        df = self.dataset.iloc[::self.row_number, :] \
            .assign(**{**{f'n_k_{i}': lambda x: (x.n_k == i).astype(int) for i in range(1, 8)},
                       **{f'n_1k_{i}': lambda x: (x.n_1k == i).astype(int) for i in range(1, 8)}}) \
            .drop(['n_k', 'n_1k'], axis=1)

        # Set currents id_k1 and iq_k1 as targets
        target_cols = ['id_k1', 'iq_k1']

        # Set remaining data as inputs
        input_cols = [c for c in df if c not in target_cols]

        # Initialize K-Fold cross validator
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=2020)

        # Preprocess the data
        ss_y = StandardScaler().fit(df[target_cols])
        df = pd.DataFrame(StandardScaler().fit_transform(df),
                          columns=df.columns)  # actually methodically unsound, but data is large enough

        # Set input and target values
        X, y = df[input_cols].values, df[target_cols].values

        return [cv, ss_y, X, y]
