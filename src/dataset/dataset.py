# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Dataset:

    # Import data upon instance creation
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.reduced_data = self.data.iloc[::1000, :]

    # Get data dimensions
    def dimensions(self):
        return self.data.shape

    # Display the first few rows of imported dataset
    def head(self):
        return self.data.head()

    # Plots element vector at time k and k-1
    # def plot_element_vector(self):
    #     # Plot n_k and n_1k
    #     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #     for c, ax in zip(['n_k', 'n_1k'], axes.flatten()):
    #         sns.countplot(x=c, data=self.data, palette="ch:.25", ax=ax)
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

    # Plot correlation matrix
    def plot_correlation_matrix(self):
        fig = plt.figure(figsize=(20, 20))
        canvas = FigureCanvas(fig)

        corr = self.reduced_data.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})

        # Add heading to the plot
        fig.suptitle('Correlation matrix', fontsize=12)

        return canvas
