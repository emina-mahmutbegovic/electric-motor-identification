# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import (QLabel, QMessageBox, QVBoxLayout,
                             QWidget, QTabWidget, QPushButton, 
                             QComboBox, QHBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt

from src.linear_regression.linear_regression import LinearRegressionModel
from src.ui.style.style import generate_label_stylesheet, generate_button_stylesheet, set_background_image, generate_combobox_stylesheet
from src.util.util import save_file

class LinearRegressionView:
    def __init__(self, parent):
        self.basic_training = None
        self.parent = parent
        self.label = None
        self.basic_training_label = None

        self.init_ui()

    def init_ui(self):
        # Set background
        set_background_image(self.parent)

        tab_widget = QTabWidget()
        self.parent.setCentralWidget(tab_widget)

        # Init basic training tab
        self.init_basic_training_tab()

        tab_widget.addTab(self.basic_training, "Basic Training")

    # Tab 1: Basic Training
    def init_basic_training_tab(self):
        self.basic_training = QWidget()
        basic_training_layout = QVBoxLayout()
        self.basic_training.setLayout(basic_training_layout)

        # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        basic_training_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        form_layout = QVBoxLayout()
        hbox_layout.addLayout(form_layout)

        self.basic_training_label = QLabel("Train linear regression model.")
        self.basic_training_label.setStyleSheet(generate_label_stylesheet("bold", "white"))
        self.basic_training_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        form_layout.addWidget(self.basic_training_label)

        # Define basic training button
        basic_training_button = QPushButton("Start Training", self.parent)
        basic_training_button.setStyleSheet(generate_button_stylesheet())
        basic_training_button.clicked.connect(lambda: self.train_linear_regression_model())

        form_layout.addWidget(basic_training_button, alignment=Qt.AlignCenter)

        for widget in self.basic_training.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def train_linear_regression_model(self):
        # Create LinearRegression model
        linear_regression = LinearRegressionModel(self.parent.dataset)

        # Create variable for scores
        scores_formatted = None

        # Train model and predict scores
        self.basic_training_label.setText("Training of linear regression model in progress...")

        # Train model
        scores = linear_regression.train()

        # Format scores output
        scores_formatted = linear_regression.format_scores_output(scores)
        # Print scores in terminal
        print(scores_formatted)

        # Prompt report download
        reply = self.parent.message_dialog.question("Download evaluation report",
                                                    "Do you want to download evaluation report?")
        if reply == QMessageBox.Yes:
            # Save report to a file
            save_file(self.parent, scores_formatted)
        else:
            self.parent.message_dialog.information("Training successfully finished!")
