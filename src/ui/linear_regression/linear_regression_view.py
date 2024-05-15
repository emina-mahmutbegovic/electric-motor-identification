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
from src.ui.enum.linear_regression_options import LinearRegressionOptions
from src.ui.style.style import generate_label_stylesheet, generate_button_stylesheet, set_background_image, generate_combobox_stylesheet
from src.util.util import save_file
from src.dataset.delayed_input import Regressor


class LinearRegressionView:
    def __init__(self, parent):
        self.hyperparameter_tuning = None
        self.basic_training = None
        self.parent = parent
        self.label = None
        self.basic_training_label = None
        self.hyperparameter_tuning_label = None
        self.regressor_combo = None

        self.selected_option = Regressor.NONE

        self.init_ui()

    def init_ui(self):
        # Set background
        set_background_image(self.parent)

        tab_widget = QTabWidget()
        self.parent.setCentralWidget(tab_widget)

        # Init basic training tab
        self.init_basic_training_tab()

        # Init hyperparemeter tuning tab
        self.init_hyperparameter_tuning_tab()

        tab_widget.addTab(self.basic_training, "Basic Training")
        #tab_widget.addTab(self.hyperparameter_tuning, "Hyperparameter Tuning")

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

        self.basic_training_label = QLabel("Train linear regression model with default parameters. \n To begin, please choose regressor:")
        self.basic_training_label.setStyleSheet(generate_label_stylesheet("bold", "white"))
        self.basic_training_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        form_layout.addWidget(self.basic_training_label)

        # Define basic training button
        basic_training_button = QPushButton("Start Training", self.parent)
        basic_training_button.setStyleSheet(generate_button_stylesheet())
        basic_training_button.clicked.connect(lambda: self.train_linear_regression_model(
            LinearRegressionOptions.BASIC_TRAINING.value))
        
        # Add regressor combobox
        self.regressor_combo = QComboBox(self.parent)
        self.regressor_combo.addItem("None", Regressor.NONE)
        self.regressor_combo.addItem("u(k-1)", Regressor.SINGLE_INPUT)
        self.regressor_combo.addItem("u(k-1), u(k-2)", Regressor.DOUBLE_INPUT)
        self.regressor_combo.addItem("u(k-3), u(k-5)", Regressor.THREE_FIVE)
        self.regressor_combo.addItem("u(k-2), u(k-3), u(k-4), u(k-5)", Regressor.TWO_THREE_FOUR_FIVE)
        self.regressor_combo.addItem("u(k-3), u(k-4), u(k-5)", Regressor.THREE_FOUR_FIVE)
        self.regressor_combo.addItem("u(k-7), u(k-11)", Regressor.SEVEN_ELEVEN)
        self.regressor_combo.addItem("u(k-6), u(k-8), u(k-15)", Regressor.SIX_EIGHT_FIFTEEN)
        self.regressor_combo.addItem("u(k-2), u(k-10), u(k-20)", Regressor.TWO_TEN_TWELVE)
        form_layout.addWidget(self.regressor_combo)

        # Connect the signal of selection change to the slot function
        self.regressor_combo.currentIndexChanged.connect(self.selection_changed)

        form_layout.addWidget(basic_training_button, alignment=Qt.AlignCenter)

        for widget in self.basic_training.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def selection_changed(self, index):
        # Get the currently selected option
        self.selected_option = self.regressor_combo.currentData()
        print("Selected option:", self.selected_option)


    # Tab 2: Hyperparameter tuning
    def init_hyperparameter_tuning_tab(self):
        self.hyperparameter_tuning = QWidget()
        hyperparameter_tuning_layout = QVBoxLayout()
        self.hyperparameter_tuning.setLayout(hyperparameter_tuning_layout)

          # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        hyperparameter_tuning_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        form_layout = QVBoxLayout()
        hbox_layout.addLayout(form_layout)

        self.hyperparameter_tuning_label = QLabel("Train linear regression model with hyperparameter tuning. \n To begin, please choose regressor:")
        self.hyperparameter_tuning_label.setStyleSheet(
            generate_label_stylesheet("bold", "white"))
        self.hyperparameter_tuning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        form_layout.addWidget(self.hyperparameter_tuning_label)

        # Define hyperparameter tuning button
        hyperparameter_tuning_button = QPushButton("Start Training", self.parent)
        hyperparameter_tuning_button.setStyleSheet(generate_button_stylesheet())
        hyperparameter_tuning_button.clicked.connect(
            lambda: self.train_linear_regression_model(LinearRegressionOptions.HYPERPARAMETER_TUNING.value))

        # Add regressor combobox
        self.regressor_combo_box = QComboBox(self.parent)
        self.regressor_combo_box.addItem("None", Regressor.NONE)
        self.regressor_combo_box.addItem("u(k-1)", Regressor.SINGLE_INPUT)
        self.regressor_combo_box.addItem("u(k-1), u(k-2)", Regressor.DOUBLE_INPUT)
        # self.regressor_combo_box.addItem("u(k-1), y(k-1)", Regressor.THREE_FIVE)

        # Connect the signal of selection change to the slot function
        self.regressor_combo_box.currentIndexChanged.connect(self.selection_changed)

        form_layout.addWidget(self.regressor_combo_box)
        form_layout.addWidget(hyperparameter_tuning_button, alignment=Qt.AlignmentFlag.AlignCenter)

        for widget in self.hyperparameter_tuning.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def train_linear_regression_model(self, option):
        # Create LinearRegression model
        linear_regression = LinearRegressionModel(self.parent.dataset, self.selected_option)

        # Create variable for scores
        scores_formatted = None

        # Train model and predict scores
        if option == LinearRegressionOptions.BASIC_TRAINING.value:
            self.basic_training_label.setText("Training of linear regression model in progress...")

            # Train model
            scores = linear_regression.train()

            # Format scores output
            scores_formatted = linear_regression.format_scores_output(scores)
            # Print scores in terminal
            print(scores_formatted)

            # Predict results
            linear_regression.predict()
        elif option == LinearRegressionOptions.HYPERPARAMETER_TUNING.value:
            self.hyperparameter_tuning_label.setText("Training of linear regression model in progress...")

            # Train model
            scores = linear_regression.hyper_parameter_tuning_train()

            # Format scores output
            scores_formatted = linear_regression.format_scores_output(scores)
            # Print scores in terminal
            print(scores_formatted)
            # Plot result
            linear_regression.hyperparameter_tuning_predict()
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
