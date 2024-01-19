# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton


class NeuralNetworkDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Configuration")

        # Main layout
        layout = QVBoxLayout()

        # Batch size
        self.batch_size_edit = QLineEdit(self)
        layout.addLayout(
            self.create_form_row("Batch Size:", self.batch_size_edit))

        # Epochs
        self.epochs_edit = QLineEdit(self)
        layout.addLayout(
            self.create_form_row("Epochs:", self.epochs_edit))

        # Hidden layer architecture
        self.hidden_layer_arch_edit = QLineEdit(self)
        layout.addLayout(
            self.create_form_row("Hidden Layer Architecture (e.g., 64-32-16):", self.hidden_layer_arch_edit))

        # Hidden layer activation function
        self.hidden_layer_activation_combo = QComboBox(self)
        self.hidden_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        layout.addLayout(self.create_form_row("Hidden Layer Activation Function:", self.hidden_layer_activation_combo))

        # Output layer activation function
        self.output_layer_activation_combo = QComboBox(self)
        self.output_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        layout.addLayout(self.create_form_row("Output Layer Activation Function:", self.output_layer_activation_combo))

        # Optimizer
        self.optimizer_combo = QComboBox(self)
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop", "adamax"])
        layout.addLayout(self.create_form_row("Optimizer:", self.optimizer_combo))

        # Loss function
        self.loss_combo = QComboBox(self)
        self.loss_combo.addItems(["binary_crossentropy", "categorical_crossentropy", "mse"])
        layout.addLayout(self.create_form_row("Loss Function:", self.loss_combo))

        # Submit button
        submit_button = QPushButton("Submit", self)
        submit_button.clicked.connect(self.on_submit)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def create_form_row(self, label_text, widget):
        row = QHBoxLayout()
        label = QLabel(label_text)
        row.addWidget(label)
        row.addWidget(widget)
        return row

    def on_submit(self):
        # Process the inputs here
        batch_size = self.batch_size_edit.text()
        epochs = self.epochs_edit.text()
        hidden_architecture = self.hidden_layer_arch_edit.text()
        hidden_activation = self.hidden_layer_activation_combo.currentText()
        output_activation = self.output_layer_activation_combo.currentText()
        optimizer = self.optimizer_combo.currentText()
        loss = self.loss_combo.currentText()

        print("Configuration:")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Hidden Layer Architecture: {hidden_architecture}")
        print(f"Hidden Layer Activation: {hidden_activation}")
        print(f"Output Layer Activation: {output_activation}")
        print(f"Optimizer: {optimizer}")
        print(f"Loss: {loss}")

        self.accept()
