# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import re

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QLabel, QMessageBox,
                             QVBoxLayout, QWidget, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QSpacerItem,
                             QSizePolicy, QTabWidget)
from keras.src.utils import plot_model

from src.neural_network.neural_network import NeuralNetworkShape, NeuralNetworkActivation, \
    NeuralNetworkLossAndOptimizer, NeuralNetwork
from src.ui.style.style import set_background_image, generate_label_stylesheet, generate_button_stylesheet, \
    generate_combobox_stylesheet, generate_line_edit_stylesheet, green_theme_color
from src.util.util import save_history_reports_csv_file, save_evaluation_reports_csv_file


def create_form_row(label_text, widget):
    row = QHBoxLayout()
    label = QLabel(label_text)
    row.addWidget(label)
    row.addWidget(widget)
    return row


def check_regex(text):
    pattern = re.compile(r'^\d+(-\d+)*$')

    return pattern.match(text)


neural_network_model_path = 'src/ui/neural_network/model_plot.png'


class NeuralNetworkView:
    def __init__(self, parent):
        self.parent = parent

        self.configuration_tab_central_widget = None
        self.display_tab_central_widget = None
        self.form_layout = None

        self.loss_combo = None
        self.optimizer_combo = None
        self.output_layer_activation_combo = None
        self.hidden_layer_activation_combo = None
        self.hidden_layer_arch_edit = None

        self.epochs_edit = None
        self.batch_size_edit = None

        self.epochs = None
        self.batch_size = None

        self.built_neural_network_model = None
        self.neural_network = None

        self.neural_network_display_label = None

        self.init_ui()

    def init_ui(self):
        # Set background
        set_background_image(self.parent)

        # Init tab widget
        tab_widget = QTabWidget()
        self.parent.setCentralWidget(tab_widget)

        # Init Neural Network Configuration Tab
        self.init_configuration_tab()

        # Init Neural Network Display Tab
        self.init_display_tab()

        tab_widget.addTab(self.configuration_tab_central_widget, "Neural Network Configuration")
        tab_widget.addTab(self.display_tab_central_widget, "Neural Network Display")

        # Connect the currentChanged signal to the on_tab_changed slot
        tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self):
        self.build_neural_network()

        if self.display_tab_central_widget is not None and self.built_neural_network_model is not None:
            self.generate_neural_network_plot()

            pixmap = QPixmap(neural_network_model_path)
            self.neural_network_display_label.setPixmap(pixmap)

    def init_configuration_tab(self):
        # Central widget and layout
        self.configuration_tab_central_widget = QWidget()
        central_layout = QVBoxLayout(
            self.configuration_tab_central_widget)  # This is the main layout for the central widget
        self.configuration_tab_central_widget.setLayout(central_layout)

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

    # Init Neural Network Configuration Form
    def init_form_layout(self):
        # Batch size
        self.batch_size_edit = QLineEdit(self.parent)
        # Set default value
        self.batch_size_edit.setText('1')
        self.form_layout.addLayout(
            create_form_row("Batch Size:", self.batch_size_edit))

        # Epochs
        self.epochs_edit = QLineEdit(self.parent)
        # Set default value
        self.epochs_edit.setText('1')
        self.form_layout.addLayout(
            create_form_row("Epochs:", self.epochs_edit))

        # Hidden layer architecture
        self.hidden_layer_arch_edit = QLineEdit(self.parent)
        # Set default value
        self.hidden_layer_arch_edit.setText('1')
        self.form_layout.addLayout(
            create_form_row("Hidden Layer Architecture (e.g., 64-32-16):", self.hidden_layer_arch_edit))

        # Hidden layer activation function
        self.hidden_layer_activation_combo = QComboBox(self.parent)
        self.hidden_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        self.form_layout.addLayout(
            create_form_row("Hidden Layer Activation Function:", self.hidden_layer_activation_combo))

        # Output layer activation function
        self.output_layer_activation_combo = QComboBox(self.parent)
        self.output_layer_activation_combo.addItems(["linear", "relu", "sigmoid", "softmax", "tanh"])
        self.form_layout.addLayout(
            create_form_row("Output Layer Activation Function:", self.output_layer_activation_combo))

        # Optimizer
        self.optimizer_combo = QComboBox(self.parent)
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop"])
        self.form_layout.addLayout(create_form_row("Optimizer:", self.optimizer_combo))

        # Loss function
        self.loss_combo = QComboBox(self.parent)
        self.loss_combo.addItems(["binary_crossentropy", "categorical_crossentropy", "mse"])
        self.form_layout.addLayout(create_form_row("Loss Function:", self.loss_combo))

        # Modify the widgets here to set text color and font
        for widget in self.configuration_tab_central_widget.findChildren(QLabel):
            widget.setStyleSheet(generate_label_stylesheet("bold", "white"))

        for widget in self.configuration_tab_central_widget.findChildren(QLineEdit):
            widget.setStyleSheet(generate_line_edit_stylesheet())

        for widget in self.configuration_tab_central_widget.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Submit button
        submit_button = QPushButton("Start Training", self.parent)
        submit_button.setStyleSheet(generate_button_stylesheet())
        submit_button.clicked.connect(lambda: self.train_neural_network_model())

        self.form_layout.addWidget(submit_button)

    def init_display_tab(self):
        # Central widget and layout
        self.display_tab_central_widget = QWidget()
        central_layout = QVBoxLayout(
            self.display_tab_central_widget)  # This is the main layout for the central widget
        self.display_tab_central_widget.setLayout(central_layout)

        # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        central_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # The form layout
        display_layout = QVBoxLayout()
        hbox_layout.addLayout(display_layout)

        # Load and display the image
        self.neural_network_display_label = QLabel()
        pixmap = QPixmap(neural_network_model_path)
        self.neural_network_display_label.setPixmap(pixmap)
        display_layout.addWidget(self.neural_network_display_label)

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def build_neural_network(self):
        # Get neural network configuration
        self.batch_size = self.batch_size_edit.text()
        self.epochs = self.epochs_edit.text()

        # Check if empty
        if self.batch_size is None or self.epochs is None:
            self.parent.message_dialog.error("Fields cannot be empty")
            return

        # Check hidden architecture input
        hidden_architecture = self.hidden_layer_arch_edit.text()
        if hidden_architecture is None:
            self.parent.message_dialog.error("Hidden layer architecture cannot be empty")
            return
        if not check_regex(hidden_architecture):
            self.parent.message_dialog.error("Please follow the pattern for hidden layer architecture")
            return

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
        self.neural_network = NeuralNetwork(shape, activation, loss_and_optimizer, self.parent.transformed_dataset)

        # Build model
        self.built_neural_network_model = self.neural_network.build_model()

    def train_neural_network_model(self):
        # Build neural network model
        self.build_neural_network()

        # Compile model
        self.neural_network.compile_model()

        try:
            # Train model
            history_reports = self.neural_network.train(int(self.epochs), int(self.batch_size))

        except Exception as e:
            self.parent.message_dialog.error(f"Error while training neural network: {e}")
        else:
            # Evaluate model
            eval_reports = self.neural_network.evaluate()

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

    def generate_neural_network_plot(self):
        plot_model(self.built_neural_network_model, to_file=neural_network_model_path, show_shapes=True,
                   show_layer_names=True, show_layer_activations=True, show_trainable=True)
