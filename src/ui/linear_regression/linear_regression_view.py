# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import (QDialog, QLabel, QMessageBox,
                             QVBoxLayout, QWidget, QTabWidget, QPushButton)
from PyQt5.QtCore import Qt

from src.linear_regression.linear_regression import LinearRegressionModel
from src.ui.enum.linear_regression_options import LinearRegressionOptions
from src.ui.style.style import generate_label_stylesheet, generate_button_stylesheet, set_background_image
from src.util.util import save_file


class LinearRegressionView:
    def __init__(self, parent):
        self.parent = parent
        self.label = None
        self.basic_training_label = None
        self.hyperparameter_tuning_label = None

        self.init_ui()

    def init_ui(self):
        # Set background
        set_background_image(self.parent)

        tab_widget = QTabWidget()
        self.parent.setCentralWidget(tab_widget)

        # Tab 1: Basic Training
        basic_training = QWidget()
        basic_training_layout = QVBoxLayout()
        basic_training.setLayout(basic_training_layout)

        self.basic_training_label = QLabel("Train linear regression model with default parameters.")
        self.basic_training_label.setStyleSheet(generate_label_stylesheet("bold", "white"))
        self.basic_training_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        basic_training_layout.addWidget(self.basic_training_label)

        # Define basic training button
        basic_training_button = QPushButton("Start Training", self.parent)
        basic_training_button.setStyleSheet(generate_button_stylesheet())
        basic_training_button.clicked.connect(lambda: self.train_linear_regression_model(
            LinearRegressionOptions.BASIC_TRAINING.value))
        basic_training_layout.addWidget(basic_training_button)

        # Tab 2: Hyperparameter tuning
        hyperparameter_tuning = QWidget()
        hyperparameter_tuning_layout = QVBoxLayout()
        hyperparameter_tuning.setLayout(hyperparameter_tuning_layout)

        self.hyperparameter_tuning_label = QLabel("Train linear regression model with hyperparameter tuning.")
        self.hyperparameter_tuning_label.setStyleSheet(
            generate_label_stylesheet("bold", "white"))
        self.hyperparameter_tuning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hyperparameter_tuning_layout.addWidget(self.hyperparameter_tuning_label)

        # Define hyperparameter tuning button
        hyperparameter_tuning_button = QPushButton("Start Training", self.parent)
        hyperparameter_tuning.setStyleSheet(generate_button_stylesheet())
        hyperparameter_tuning_button.clicked.connect(
            lambda: self.train_linear_regression_model(LinearRegressionOptions.HYPERPARAMETER_TUNING.value))
        hyperparameter_tuning_layout.addWidget(hyperparameter_tuning_button)

        tab_widget.addTab(basic_training, "Basic Training")
        tab_widget.addTab(hyperparameter_tuning, "Hyperparameter Tuning")

    def train_linear_regression_model(self, option):
        # Create LinearRegression model
        linear_regression = LinearRegressionModel(self.parent.transformed_dataset)

        # Create variable for scores
        scores_formatted = None

        # Train model and predict scores
        if option == LinearRegressionOptions.BASIC_TRAINING.value:
            self.basic_training_label.setText("Training of linear regression model in progress...")

            # Train model
            scores = linear_regression.train_and_predict()

            # Format scores output
            scores_formatted = linear_regression.format_scores_output(scores)
            # Print scores in terminal
            print(scores_formatted)
            # Display result
            self.basic_training_label.setText(scores_formatted)
        elif option == LinearRegressionOptions.HYPERPARAMETER_TUNING.value:
            self.hyperparameter_tuning_label.setText("Training of linear regression model in progress...")

            # Train model
            scores = linear_regression.hyper_parameter_tuning()

            # Format scores output
            scores_formatted = linear_regression.format_scores_output(scores)
            # Print scores in terminal
            print(scores_formatted)
            # Display result
            self.basic_training_label.setText(scores_formatted)
        else:
            self.parent.message_dialog.error("Status unknown")

        # Prompt report download
        reply = self.parent.message_dialog.question("Download evaluation report",
                                                    "Do you want to download evaluation report?")
        if reply == QMessageBox.Yes:
            # Save report to a file
            save_file(self.parent, scores_formatted)
        else:
            self.parent.message_dialog.information("Training successfully finished!")

