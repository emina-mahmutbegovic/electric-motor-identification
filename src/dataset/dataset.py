# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Dataset:

    # Import data upon instance creation
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.reduced_data = self.data.iloc[:2000]

        # Set currents id_k1 and iq_k1 as targets
        self.target_cols = ['id_k1', 'iq_k1']

        # Set currents id_k and iq_k as inputs
        self.input_cols = ['id_k', 'iq_k']

        self.input_id = self.reduced_data['id_k'].values
        self.input_iq = self.reduced_data['iq_k'].values

        self.output_id_k1 = self.reduced_data['id_k1'].values
        self.output_iq_k1 = self.reduced_data['iq_k1'].values

        # Set input and target values
        self.X, self.y = self.reduced_data[self.input_cols].values, self.reduced_data[self.target_cols].values

    # Get data dimensions
    def dimensions(self):
        return self.data.shape

    # Display the first few rows of imported dataset
    def head(self):
        return self.data.head()

    # Plots element vector at time k and k-1
    def plot_element_vector(self):
        fig = plt.figure()
        canvas = FigureCanvas(fig)

        # Your plotting code
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        for c, ax in zip(['n_k', 'n_1k'], axes.flatten()):
            sns.countplot(x=c, data=self.data, palette="ch:.25", ax=ax)

        # Add heading to the plot
        fig.suptitle('Element vector plot at time k and k-1', fontsize=12)

        return canvas
    
    def plot_results(self, var1, var2, label1='Variable id_k', label2='Variable iq_k'): 
        time = np.linspace(0, 10, 100)  # 100 time points from 0 to 10
        # Create a plot
        plt.figure(figsize=(10, 5))  # Set the figure size

        # Plot both variables
        plt.plot(time, var1[:100], label=label1, color='blue')
        plt.plot(time, var2[:100], label=label2, color='red')

        # Adding title and labels
        plt.title('The first 100 target variables over time')
        plt.xlabel('Time')
        plt.ylabel('Value')

        # Add a legend to the plot
        plt.legend()

        # Show the plot
        plt.show()

    # Plot histogram against element vector
    def plot_histogram_nk(self):
        fig = plt.figure(figsize=(20, 20))
        canvas = FigureCanvas(fig)

        analyzed_cols = [c for c in self.data if c != 'n_k']
        unique_elem_vecs = self.data['n_k'].nunique()

        axes = fig.subplots(nrows=unique_elem_vecs, ncols=len(analyzed_cols), sharex='col')

        for k, df in self.reduced_data.groupby('n_k'):
            for i, c in enumerate(analyzed_cols):
                sns.histplot(df[c], ax=axes[k - 1, i])
                if i == 0:
                    axes[k - 1, i].set_ylabel(f'n_k = {k}')

        plt.tight_layout()

        # Add heading to the plot
        fig.suptitle('Histogram against element vector', fontsize=12)

        return canvas

    # Describe rotor angle column
    def describe_rotor_angle_column(self):
        self.reduced_data['epsilon_k'].describe()

    # Transform data
    def transform(self):
        # Add sine and cosine of the rotor angle and the current vector norm
        return self.data.assign(sin_eps_k=lambda df: np.sin(df.epsilon_k),
                                cos_eps_k=lambda df: np.cos(df.epsilon_k),
                                i_norm=lambda df: np.sqrt(df.id_k ** 2 + df.iq_k ** 2)).drop('epsilon_k', axis=1)
    
    # Split data to inputs and targets
    def split_data(self, data):
        # Set input and target values
        X, y = data[self.input_cols].values, data[self.target_cols].values
        
        return [X, y]
    
    def pearson(self):
        # Check if data is existing
        if(self.X is None or self.y is None or not self.input_cols or not self.target_cols):
            print("ERROR: Data not preprocessed. Cannot calculate Pearson's coefficient.")
            return

        results_dict = {}

        for i in range(self.X.shape[1]):  # For each input feature
            for j in range(self.y.shape[1]):  # For each output feature
                corr, p_val = pearsonr(self.X[:, i], self.y[:, j])

                # Update dictionary
                if self.input_cols[i] not in results_dict:
                    results_dict[self.input_cols[i]] = {}

                # Update dictionary
                results_dict[self.input_cols[i]][self.target_cols[j]] = {"correlation": corr, "p_value": p_val}

        return {
            'target_cols': self.target_cols,
            'results_dict': results_dict
        }
    
    def spearman(self):
        # Check if data is existing
        if(self.X is None or self.y is None or not self.input_cols or not self.target_cols):
            print("ERROR: Data not preprocessed. Cannot calculate Spearman's coefficient.")
            return

        results_dict = {}

        for i in range(self.X.shape[1]):  # For each input feature
            for j in range(self.y.shape[1]):  # For each output feature
                corr, p_val = spearmanr(self.X[:, i], self.y[:, j])

                # Update dictionary
                if self.input_cols[i] not in results_dict:
                    results_dict[self.input_cols[i]] = {}

                # Update dictionary
                results_dict[self.input_cols[i]][self.target_cols[j]] = {"correlation": corr, "p_value": p_val}

        return {
            'target_cols': self.target_cols,
            'results_dict': results_dict
        }

    # Plot correlation map
    def plot_correlation_map(self, correlations_dict):    
        # Extract outputs
        output1 = correlations_dict['target_cols'][0]
        output2 = correlations_dict['target_cols'][1]

        # Extract Pearson correlations dict values
        correlations_dict_values = correlations_dict['results_dict']

        # Transform dict into an array of correlations
        correlations_output1 = []
        correlations_output2 = []
        input_labels = list(correlations_dict_values.keys())

        for input in input_labels:
            correlations_output1.append(correlations_dict_values[input][output1]['correlation'])
            correlations_output2.append(correlations_dict_values[input][output2]['correlation'])
            
        fig, ax = plt.subplots(figsize=(20, 20))
        canvas = FigureCanvas(fig)


        # Set position of bar on X axis
        barWidth = 0.4
        r1 = np.arange(len(correlations_output1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.bar(r1, correlations_output1, color='blue', width=barWidth, edgecolor='grey', label=output1)
        ax.bar(r2, correlations_output2, color='red', width=barWidth, edgecolor='grey', label=output2)

        # Add xticks on the middle of the group bars
        ax.set_xlabel('Input Features', fontweight='bold')
        ax.set_ylabel('Correlation', fontweight='bold')
        ax.set_xticks([r + barWidth/2 for r in range(len(correlations_output1))])
        ax.set_xticklabels(input_labels, rotation=90)
        ax.set_title('Data Correlation')

        # Create legend & Show graphic
        ax.legend()
        return canvas
