# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import (QDialog, QLabel, QMessageBox,
                             QVBoxLayout, QWidget, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QSpacerItem,
                             QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt

from src.neural_network.neural_network import NeuralNetworkShape, NeuralNetworkActivation, \
    NeuralNetworkLossAndOptimizer, NeuralNetwork
from src.ui.style.style import set_background_image, generate_label_stylesheet, generate_button_stylesheet, \
    generate_combobox_stylesheet, generate_line_edit_stylesheet
from src.util.util import save_file, save_history_reports_csv_file, save_evaluation_reports_csv_file


class NeuralNetworkView:
    def __init__(self, parent):
        self.central_widget = None
        self.form_layout = None
        self.parent = parent
        self.start_training_button = None
        self.loss_combo = None
        self.optimizer_combo = None
        self.output_layer_activation_combo = None
        self.hidden_layer_activation_combo = None
        self.hidden_layer_arch_edit = None
        self.epochs_edit = None
        self.batch_size_edit = None

        self.init_ui()

    def init_ui(self):
        # Set background
        set_background_image(self.parent)

        configure_neural_network_tab = QTabWidget()
        self.parent.setCentralWidget(configure_neural_network_tab)

        # Central widget and layout
        self.central_widget = QWidget()
        central_layout = QVBoxLayout(self.central_widget)  # This is the main layout for the central widget
        self.central_widget.setLayout(central_layout)

        # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        central_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # The form layout
        self.form_layout = QVBoxLayout()
        hbox_layout.addLayout(self.form_layout)

        self.init_form_layout()

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        configure_neural_network_tab.addTab(self.central_widget, "Neural Network Configuration")

    def init_form_layout(self):
        # Batch size
        self.batch_size_edit = QLineEdit(self.parent)
        self.form_layout.addLayout(
            self.create_form_row("Batch Size:", self.batch_size_edit))

        # Epochs
        self.epochs_edit = QLineEdit(self.parent)
        self.form_layout.addLayout(
            self.create_form_row("Epochs:", self.epochs_edit))

        # Hidden layer architecture
        self.hidden_layer_arch_edit = QLineEdit(self.parent)
        self.form_layout.addLayout(
            self.create_form_row("Hidden Layer Architecture (e.g., 64-32-16):", self.hidden_layer_arch_edit))

        # Hidden layer activation function
        self.hidden_layer_activation_combo = QComboBox(self.parent)
        self.hidden_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        self.form_layout.addLayout(
            self.create_form_row("Hidden Layer Activation Function:", self.hidden_layer_activation_combo))

        # Output layer activation function
        self.output_layer_activation_combo = QComboBox(self.parent)
        self.output_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        self.form_layout.addLayout(
            self.create_form_row("Output Layer Activation Function:", self.output_layer_activation_combo))

        # Optimizer
        self.optimizer_combo = QComboBox(self.parent)
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop"])
        self.form_layout.addLayout(self.create_form_row("Optimizer:", self.optimizer_combo))

        # Loss function
        self.loss_combo = QComboBox(self.parent)
        self.loss_combo.addItems(["binary_crossentropy", "categorical_crossentropy", "mse"])
        self.form_layout.addLayout(self.create_form_row("Loss Function:", self.loss_combo))

        # Modify the widgets here to set text color and font
        for widget in self.central_widget.findChildren(QLabel):
            widget.setStyleSheet(generate_label_stylesheet("bold", "white"))

        for widget in self.central_widget.findChildren(QLineEdit):
            widget.setStyleSheet(generate_line_edit_stylesheet())

        for widget in self.central_widget.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Submit button
        submit_button = QPushButton("Start Training", self.parent)
        submit_button.setStyleSheet(generate_button_stylesheet())
        submit_button.clicked.connect(lambda: self.train_neural_network_model())

        self.form_layout.addWidget(submit_button)

    def create_form_row(self, label_text, widget):
        row = QHBoxLayout()
        label = QLabel(label_text)
        row.addWidget(label)
        row.addWidget(widget)
        return row

    def train_neural_network_model(self):
        # Get neural network configuration
        batch_size = self.batch_size_edit.text()
        epochs = self.epochs_edit.text()

        hidden_architecture = self.hidden_layer_arch_edit.text().split('-')
        # Input and output shape is based on dataset
        shape = NeuralNetworkShape((19,), hidden_architecture, 2)

        hidden_activation = self.hidden_layer_activation_combo.currentText()
        output_activation = self.output_layer_activation_combo.currentText()
        activation = NeuralNetworkActivation(hidden_activation, output_activation)

        optimizer = self.optimizer_combo.currentText()
        loss = self.loss_combo.currentText()
        loss_and_optimizer = NeuralNetworkLossAndOptimizer(loss, optimizer)

        # Create NeuralNetwork model
        neural_network = NeuralNetwork(shape, activation, loss_and_optimizer, self.parent.transformed_dataset)

        # Build model
        neural_network.build_model()

        # Compile model
        neural_network.compile_model()

        # Train model
        history_reports = neural_network.train(int(epochs), int(batch_size))

        # Evaluate model
        eval_reports = neural_network.evaluate()

        # Download reports
        # Prompt report download
        reply = self.parent.message_dialog.question("Download reports",
                                                    "Do you want to download training and evaluation "
                                                    "report?")
        if reply == QMessageBox.Yes:
            # Save reports to csv files
            save_history_reports_csv_file(self.parent, history_reports)
            save_evaluation_reports_csv_file(self.parent, eval_reports)
        else:
            self.parent.message_dialog.information("Training successfully finished!")
