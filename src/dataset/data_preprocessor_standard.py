# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import pandas as pd
from sklearn.preprocessing import StandardScaler

class StandardDataPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset

        # Set currents id_k1 and iq_k1 as targets
        self.target_cols = ['id_k1', 'iq_k1']

        # Set remaining data as inputs
        #self.input_cols = [c for c in self.df if c not in self.target_cols]

    # Preprocess data for training by using standard scaler
    def preprocess_standard(self):
        # Preprocess the data using standard scaler
        ss_y = StandardScaler().fit(self.dataset[self.target_cols])
        df = pd.DataFrame(StandardScaler().fit_transform(self.dataset),
                          columns=self.dataset.columns)  # actually methodically unsound, but data is large enough

        # Set input currents
        #id_k_standardized = df['id_k'].values
        #iq_k_standardized = df['iq_k'].values

        return [ss_y, df]