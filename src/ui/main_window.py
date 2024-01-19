# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QToolBar, QAction,
    QStatusBar, QMessageBox, QDialog,
    QVBoxLayout, QPushButton, QWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QSize

from src.ui.dialog.neural_network_dialog import NeuralNetworkDialog
from src.ui.enum.linear_regression_options import LinearRegressionOptions
from src.dataset.dataset import Dataset
from src.linear_regression.linear_regression import LinearRegressionModel
from src.ui.dialog.linear_regression_dialog import LinearRegressionOptionDialog
from src.neural_network.neural_network import (NeuralNetworkActivation, NeuralNetworkLossAndOptimizer,
                                               NeuralNetworkShape, NeuralNetwork)
from src.ui.dialog.message_dialog import MessageDialog
from src.util.util import (save_file, save_evaluation_reports_csv_file,
                           save_history_reports_csv_file, upload_csv)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Define message dialog
        self.message_dialog = MessageDialog(self)

        # Define dataset
        self.dataset = []

        self.setWindowTitle("ML Training Application")

        self.setFixedSize(QSize(600, 400))

        self.layout = QVBoxLayout()

        # Add a QLabel to display text
        self.label = QLabel("Welcome to the ML Training Application!\n\n"
                            "To proceed, please upload dataset in CSV format.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        # Add a QPushButton
        self.upload_button = QPushButton("Upload")
        self.upload_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        self.upload_button.clicked.connect(self.upload_dataset)
        self.layout.addWidget(self.upload_button)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def upload_dataset(self):
        self.label.setText("Importing dataset... Please wait.")

        file_path = upload_csv(self)

        if file_path:
            self.dataset = Dataset(file_path).transform()

            # Clear upload button from initial layout
            # Remove the button from the layout
            self.layout.removeWidget(self.upload_button)

            # Delete the button
            self.upload_button.deleteLater()

            # Display toolbar view
            self.display_toolbar_view()

            # Update label
            self.label.setText("Welcome to the ML Training Application!\n\n"
                               "To proceed, please choose one of the model options in the toolbar.")
        else:
            self.label.setText("Welcome to the ML Training Application!\n\n"
                               "To proceed, please upload dataset in CSV format.")

    def display_toolbar_view(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        linear_regression_button = QAction("Linear Regression", self)
        linear_regression_button.setStatusTip("Train linear regression model")
        linear_regression_button.triggered.connect(self.show_linear_regression_dialog)
        toolbar.addAction(linear_regression_button)

        toolbar.addSeparator()

        neural_network_button = QAction("Neural Network", self)
        neural_network_button.setStatusTip("Train neural network model")
        neural_network_button.triggered.connect(self.show_neural_network_dialog)
        toolbar.addAction(neural_network_button)

        self.setStatusBar(QStatusBar(self))

    def show_linear_regression_dialog(self):
        self.label.setText("Training linear regression model...")

        dialog = LinearRegressionOptionDialog()
        result = dialog.exec_()

        if result == LinearRegressionOptions.BASIC_TRAINING.value:
            self.train_linear_regression_model(LinearRegressionOptions.BASIC_TRAINING.value)
        elif result == LinearRegressionOptions.HYPERPARAMETER_TUNING.value:
            self.train_linear_regression_model(LinearRegressionOptions.HYPERPARAMETER_TUNING.value)
        elif result == QDialog.Rejected:
            self.message_dialog.information("Dialog was closed without selecting an option. "
                                            "Please select an option to continue.")
        else:
            self.message_dialog.error("Status unknown. Please try again.")

    def train_linear_regression_model(self, option):
        # Create LinearRegression model
        linear_regression = LinearRegressionModel(self.dataset)

        # Create variable for scores
        scores = []

        # Train model and predict scores
        if option == LinearRegressionOptions.BASIC_TRAINING.value:
            scores = linear_regression.train_and_predict()
        elif option == LinearRegressionOptions.HYPERPARAMETER_TUNING.value:
            scores = linear_regression.hyper_parameter_tuning()

        # Format scores output
        scores_formatted = linear_regression.format_scores_output(scores)

        # Print scores in terminal
        print(scores_formatted)

        # Set result
        self.label.setText(scores_formatted)

        # Prompt report download
        reply = self.message_dialog.question("Download evaluation report", "Do you want to download evaluation report?")
        if reply == QMessageBox.Yes:
            # Save report to a file
            save_file(self, scores_formatted)
        else:
            self.message_dialog.information("Training successfully finished!")

    def show_neural_network_dialog(self):
        self.label.setText("Training neural network model...")

        dialog = NeuralNetworkDialog()

        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Extract and process submitted data
            batch_size = dialog.batch_size_edit.text()
            epochs = dialog.epochs_edit.text()

            hidden_architecture = dialog.hidden_layer_arch_edit.text().split('-')
            # Input and output shape is based on dataset
            shape = NeuralNetworkShape((19,), hidden_architecture, 2)

            hidden_activation = dialog.hidden_layer_activation_combo.currentText()
            output_activation = dialog.output_layer_activation_combo.currentText()
            activation = NeuralNetworkActivation(hidden_activation, output_activation)

            optimizer = dialog.optimizer_combo.currentText()
            loss = dialog.loss_combo.currentText()
            loss_and_optimizer = NeuralNetworkLossAndOptimizer(loss, optimizer)

            self.train_neural_network_model(int(epochs), int(batch_size), shape, activation, loss_and_optimizer)
        elif result == QDialog.Rejected:
            self.message_dialog.information("Please configure neural network to continue.")

    def train_neural_network_model(self, epochs, batch_size, shape, activation, loss_and_optimizer):
        # Create NeuralNetwork model
        neural_network = NeuralNetwork(shape, activation, loss_and_optimizer, self.dataset)

        # Build model
        neural_network.build_model()

        # Compile model
        neural_network.compile_model()

        # Train model
        history_reports = neural_network.train(self.label, epochs, batch_size)

        # Evaluate model
        eval_reports = neural_network.evaluate()

        # Download reports
        # Prompt report download
        reply = self.message_dialog.question("Download reports", "Do you want to download training and evaluation "
                                                                 "report?")
        if reply == QMessageBox.Yes:
            # Save reports to csv files
            save_history_reports_csv_file(self, history_reports)
            save_evaluation_reports_csv_file(self, eval_reports)
        else:
            self.message_dialog.information("Training successfully finished!")
