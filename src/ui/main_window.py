# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtGui import QFontMetrics, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QToolBar, QAction,
    QStatusBar, QVBoxLayout, QPushButton, QWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QSize

from src.ui.dataset.dataset_view import DatasetView
from src.dataset.dataset import Dataset
from src.ui.dialog.message_dialog import MessageDialog
from src.ui.linear_regression.linear_regression_view import LinearRegressionView
from src.ui.neural_network.neural_network_view import NeuralNetworkView
from src.ui.style.style import set_background_image, generate_label_stylesheet, generate_button_stylesheet
from src.util.util import upload_csv, get_file_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Define message dialog
        self.upload_button = None
        self.label = None
        self.layout = None
        self.message_dialog = MessageDialog(self)

        # Define dataset
        self.dataset = []
        # Define transformed data that will be used for training
        self.transformed_dataset = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Electric Motor Identification")

        self.resize(QSize(1000, 800))

        set_background_image(self)

        self.layout = QVBoxLayout()

        # Add a QLabel to display text
        self.label = QLabel("Welcome to the Electric Motor Identification Application!\n\n"
                            "To proceed, please upload dataset in CSV format.")

        # Set label style
        self.label.setStyleSheet(generate_label_stylesheet("bold", "white"))

        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        # Add a QPushButton
        self.upload_button = QPushButton("Upload")
        self.upload_button.setStyleSheet(generate_button_stylesheet())
        self.upload_button.setIcon(QIcon(get_file_path('src/ui/assets/power-plug.png')))
        self.upload_button.setIconSize(QSize(16, 16))  # Set the size of the icon
        self.upload_button.clicked.connect(self.upload_dataset)
        self.layout.addWidget(self.upload_button)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def upload_dataset(self):
        self.label.setText("Importing dataset... Please wait.")

        file_path = upload_csv(self)

        if file_path:
            self.dataset = Dataset(file_path)
            self.transformed_dataset = self.dataset.transform()

            # Clear upload button from initial layout
            # Remove the button from the layout
            self.layout.removeWidget(self.upload_button)

            # Delete the button
            self.upload_button.deleteLater()

            # Display toolbar view
            self.show_toolbar_view()

            # Update label
            self.label.setText("Welcome to the Electric Motor Identification Application!\n\n"
                               "To proceed, please choose one of the model options in the toolbar.")
        else:
            self.label.setText("Welcome to the Electric Motor Identification Application!\n\n"
                               "To proceed, please upload dataset in CSV format.")

    def show_toolbar_view(self):
        # Initialize toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        # Disable toolbar movement
        toolbar.setMovable(False)

        # Set dataset option
        dataset_button = QAction("Dataset", self)
        dataset_button.setStatusTip("Analyze dataset")
        dataset_button.triggered.connect(self.show_dataset_view)
        toolbar.addAction(dataset_button)

        toolbar.addSeparator()

        # Set linear regression option
        linear_regression_button = QAction("Linear Regression", self)
        linear_regression_button.setStatusTip("Train linear regression model")
        linear_regression_button.triggered.connect(self.show_linear_regression_view)
        toolbar.addAction(linear_regression_button)

        toolbar.addSeparator()

        # Set neural network option
        neural_network_button = QAction("Neural Network", self)
        neural_network_button.setStatusTip("Train neural network model")
        neural_network_button.triggered.connect(self.show_neural_network_view)
        toolbar.addAction(neural_network_button)

        self.setStatusBar(QStatusBar(self))

    def show_dataset_view(self):
        dataset_view = DatasetView(self)
        dataset_view.init_ui()

    def show_linear_regression_view(self):
        linear_regression_view = LinearRegressionView(self)
        linear_regression_view.init_ui()

    def show_neural_network_view(self):
        neural_network_view = NeuralNetworkView(self)
        neural_network_view.init_ui()
