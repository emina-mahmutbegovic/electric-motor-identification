# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QDialog, QPushButton)

from src.ui.enum.linear_regression_options import LinearRegressionOptions


class LinearRegressionOptionDialog(QDialog):
    def __init__(self):
        super().__init__()

        # Set up the layout
        layout = QVBoxLayout()

        # Add a label for instruction or information
        label = QLabel("Choose an option for training:")
        layout.addWidget(label)

        # Add the first option button
        self.basicTrainingButton = QPushButton("Basic Training", self)
        self.basicTrainingButton.clicked.connect(self.on_basic_training)
        layout.addWidget(self.basicTrainingButton)

        # Add the second option button
        self.hyperperameterTuningButton = QPushButton("Hyperparameter Tuning", self)
        self.hyperperameterTuningButton.clicked.connect(self.on_hyperparameter_tuning)
        layout.addWidget(self.hyperperameterTuningButton)

        # Set the dialog layout
        self.setLayout(layout)

    def on_basic_training(self):
        self.done(LinearRegressionOptions.BASIC_TRAINING.value)

    def on_hyperparameter_tuning(self):
        self.done(LinearRegressionOptions.HYPERPARAMETER_TUNING.value)
