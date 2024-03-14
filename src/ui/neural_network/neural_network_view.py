# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import re
import time

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QLabel, QMessageBox,
                             QVBoxLayout, QWidget, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QSpacerItem,
                             QSizePolicy, QTabWidget)
from keras.src.utils import plot_model

from src.neural_network.neural_network import NeuralNetworkShape, NeuralNetworkActivation, \
    NeuralNetworkLossAndOptimizer, NeuralNetwork
from src.lstm_network.lstm_network import LSTMNetwork
from src.dataset.data_preprocessor_lstm import LSTMDataPreprocessor

from src.ui.style.style import set_background_image, generate_label_stylesheet, generate_button_stylesheet, \
    generate_combobox_stylesheet, generate_line_edit_stylesheet

from src.util.util import save_history_reports_csv_file, save_evaluation_reports_csv_file, save_history_and_val_reports_csv_file
from src.util.shared import stop_neural_network_training_flag, activation_functions, loss_functions, metrics, optimizers, data_row

from src.ui.neural_network.config.neural_network_config import NeuralNetworkConfig
from src.ui.neural_network.config.lstm_config import LSTMConfig

def create_form_row(label_text, widget):
    row = QHBoxLayout()
    label = QLabel(label_text)
    row.addWidget(label)
    row.addWidget(widget)
    return row


def check_regex(text):
    pattern = re.compile(r'^\d+(-\d+)*$')

    return pattern.match(text)

export_model_path_base = 'src/ui/neural_network/'
neural_network_model_path = export_model_path_base + 'model_plot.png'
lstm_network_model_path = export_model_path_base + 'lstm_model_plot.png'

class NeuralNetworkView:
    def __init__(self, parent):
        self.parent = parent

        self.configuration_tab_central_widget = None
        self.display_tab_central_widget = None
        self.lstm_display_tab_central_widget = None
        self.form_layout = None

        self.neural_network_config = NeuralNetworkConfig()
        self.lstm_config = LSTMConfig()

        self.built_neural_network_model = None
        self.neural_network = None

        self.built_lstm_network_model = None
        self.lstm_network = None

        self.neural_network_display_label = None
        self.lstm_display_label = None

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

        # Init LSTM Tab
        self.init_lstm_tab()

        # Init LSTM Network Display Tab
        self.init_lstm_display_tab()

        tab_widget.addTab(self.configuration_tab_central_widget, "Neural Network Configuration")
        tab_widget.addTab(self.display_tab_central_widget, "Neural Network Display")
        tab_widget.addTab(self.lstm_tab_central_widget, "LSTM Network")
        tab_widget.addTab(self.lstm_display_tab_central_widget, "LSTM Network Display")

        # Connect the currentChanged signal to the on_tab_changed slot
        tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self):
        self.build_neural_network()
        self.build_and_compile_lstm_network()

        if self.display_tab_central_widget is not None and self.built_neural_network_model is not None:
            self.generate_neural_network_plot(self.built_neural_network_model, neural_network_model_path)

            pixmap = QPixmap(neural_network_model_path)
            self.neural_network_display_label.setPixmap(pixmap)
        
        if self.lstm_display_tab_central_widget is not None and self.lstm_network.model is not None:
            self.generate_neural_network_plot(self.lstm_network.model, lstm_network_model_path)

            pixmap = QPixmap(lstm_network_model_path)
            self.lstm_display_label.setPixmap(pixmap)

    # Neural Network Configuration Tab
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
        self.neural_network_form_layout = QVBoxLayout()
        hbox_layout.addLayout(self.neural_network_form_layout)

        self.init_neural_network_form_layout()

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    # Init Neural Network Configuration Form
    def init_neural_network_form_layout(self):
        # Batch size
        self.neural_network_config.batch_size_edit = QLineEdit(self.parent)
        # Set default value
        self.neural_network_config.batch_size_edit.setText('1')
        self.neural_network_form_layout.addLayout(
            create_form_row("Batch Size:", self.neural_network_config.batch_size_edit))

        # Epochs
        self.neural_network_config.epochs_edit = QLineEdit(self.parent)
        # Set default value
        self.neural_network_config.epochs_edit.setText('1')
        self.neural_network_form_layout.addLayout(
            create_form_row("Epochs:", self.neural_network_config.epochs_edit))

        # Hidden layer architecture
        self.neural_network_config.hidden_layer_arch_edit = QLineEdit(self.parent)
        # Set default value
        self.neural_network_config.hidden_layer_arch_edit.setText('1')
        self.neural_network_form_layout.addLayout(
            create_form_row("Hidden Layer Architecture (e.g., 64-32-16):", self.neural_network_config.hidden_layer_arch_edit))

        # Hidden layer activation function
        self.neural_network_config.hidden_layer_activation_combo = QComboBox(self.parent)
        self.neural_network_config.hidden_layer_activation_combo.addItems(activation_functions)
        self.neural_network_form_layout.addLayout(
            create_form_row("Hidden Layer Activation Function:", self.neural_network_config.hidden_layer_activation_combo))

        # Output layer activation function
        self.neural_network_config.output_layer_activation_combo = QComboBox(self.parent)
        self.neural_network_config.output_layer_activation_combo.addItems(activation_functions)
        self.neural_network_form_layout.addLayout(
            create_form_row("Output Layer Activation Function:", self.neural_network_config.output_layer_activation_combo))

        # Optimizer
        self.neural_network_config.optimizer_combo = QComboBox(self.parent)
        self.neural_network_config.optimizer_combo.addItems(optimizers)
        self.neural_network_form_layout.addLayout(create_form_row("Optimizer:", self.neural_network_config.optimizer_combo))

        # Loss function
        self.neural_network_config.loss_combo = QComboBox(self.parent)
        self.neural_network_config.loss_combo.addItems(loss_functions)
        self.neural_network_form_layout.addLayout(create_form_row("Loss Function:", self.neural_network_config.loss_combo))

        # Metrics
        self.neural_network_config.metrics_combo = QComboBox(self.parent)
        self.neural_network_config.metrics_combo.addItems(metrics)
        self.neural_network_form_layout.addLayout(create_form_row("Metrics:", self.neural_network_config.metrics_combo))

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

        self.neural_network_form_layout.addWidget(submit_button)

    # Neural Network Display Tab
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

    # LSTM Network Display Tab
    def init_lstm_display_tab(self):
        # Central widget and layout
        self.lstm_display_tab_central_widget = QWidget()
        central_layout = QVBoxLayout(
            self.lstm_display_tab_central_widget)  # This is the main layout for the central widget
        self.lstm_display_tab_central_widget.setLayout(central_layout)

        # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        central_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # The form layout
        display_layout = QVBoxLayout()
        hbox_layout.addLayout(display_layout)

        # Load and display the image
        self.lstm_display_label = QLabel()
        pixmap = QPixmap(lstm_network_model_path)
        self.lstm_display_label.setPixmap(pixmap)
        display_layout.addWidget(self.lstm_display_label)

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
    
    # LSTM tab
    def init_lstm_tab(self):
        # Central widget and layout
        self.lstm_tab_central_widget = QWidget()
        central_layout = QVBoxLayout(
            self.lstm_tab_central_widget)  # This is the main layout for the central widget
        self.lstm_tab_central_widget.setLayout(central_layout)

        # Horizontal layout for centering the content
        hbox_layout = QHBoxLayout()
        central_layout.addLayout(hbox_layout)

        # Add horizontal spacers to center the form
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # The form layout
        self.lstm_form_layout = QVBoxLayout()
        hbox_layout.addLayout(self.lstm_form_layout)

        self.init_lstm_form_layout()

        # Add another horizontal spacer for symmetry
        hbox_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    # Init LSTM Network Configuration Form
    def init_lstm_form_layout(self):
        # Batch size
        self.lstm_config.batch_size_edit = QLineEdit(self.parent)
        # Set default value
        self.lstm_config.batch_size_edit.setText('1')
        self.lstm_form_layout.addLayout(
            create_form_row("Batch Size:", self.lstm_config.batch_size_edit))

        # Epochs
        self.lstm_config.epochs_edit = QLineEdit(self.parent)
        # Set default value
        self.lstm_config.epochs_edit.setText('1')
        self.lstm_form_layout.addLayout(
            create_form_row("Epochs:", self.lstm_config.epochs_edit))
        
        # Num of units
        self.lstm_config.num_of_units_edit = QLineEdit(self.parent)
        # Set default value
        self.lstm_config.num_of_units_edit.setText('1')
        self.lstm_form_layout.addLayout(
            create_form_row("Number of Units:", self.lstm_config.num_of_units_edit))
        
        # Num of look back steps
        self.lstm_config.look_back_edit = QLineEdit(self.parent)
        # Set default value
        self.lstm_config.look_back_edit.setText('1')
        self.lstm_form_layout.addLayout(
            create_form_row("Number of Look Back Steps:", self.lstm_config.look_back_edit))

        # Output activation function
        self.lstm_config.activation_function = QComboBox(self.parent)
        self.lstm_config.activation_function.addItems(activation_functions)
        self.lstm_form_layout.addLayout(
            create_form_row("Activation Function:", self.lstm_config.activation_function))

        # Optimizer
        self.lstm_config.optimizer_combo = QComboBox(self.parent)
        self.lstm_config.optimizer_combo.addItems(optimizers)
        self.lstm_form_layout.addLayout(create_form_row("Optimizer:", self.lstm_config.optimizer_combo))

        # Loss function
        self.lstm_config.loss_combo = QComboBox(self.parent)
        self.lstm_config.loss_combo.addItems(loss_functions)
        self.lstm_form_layout.addLayout(create_form_row("Loss Function:", self.lstm_config.loss_combo))

        # Metrics
        self.lstm_config.metrics_combo = QComboBox(self.parent)
        self.lstm_config.metrics_combo.addItems(metrics)
        self.lstm_form_layout.addLayout(create_form_row("Metrics:", self.lstm_config.metrics_combo))

        # Modify the widgets here to set text color and font
        for widget in self.lstm_tab_central_widget.findChildren(QLabel):
            widget.setStyleSheet(generate_label_stylesheet("bold", "white"))

        for widget in self.lstm_tab_central_widget.findChildren(QLineEdit):
            widget.setStyleSheet(generate_line_edit_stylesheet())

        for widget in self.lstm_tab_central_widget.findChildren(QComboBox):
            widget.setStyleSheet(generate_combobox_stylesheet())

        # Submit button
        submit_button = QPushButton("Start Training", self.parent)
        submit_button.setStyleSheet(generate_button_stylesheet())
        submit_button.clicked.connect(lambda: self.train_lstm_network())

        self.lstm_form_layout.addWidget(submit_button)

    def build_neural_network(self):
        # Get neural network configuration
        self.neural_network_config.batch_size = self.neural_network_config.batch_size_edit.text()
        self.neural_network_config.epochs = self.neural_network_config.epochs_edit.text()

        # Check if empty
        if self.neural_network_config.batch_size is None or self.neural_network_config.epochs is None:
            self.parent.message_dialog.error("Fields cannot be empty")
            return

        # Check hidden architecture input
        hidden_architecture = self.neural_network_config.hidden_layer_arch_edit.text()
        if hidden_architecture is None:
            self.parent.message_dialog.error("Hidden layer architecture cannot be empty")
            return
        if not check_regex(hidden_architecture):
            self.parent.message_dialog.error("Please follow the pattern for hidden layer architecture")
            return

        hidden_architecture = self.neural_network_config.hidden_layer_arch_edit.text().split('-')
        # Input and output shape is based on dataset
        shape = NeuralNetworkShape((19,), hidden_architecture, 2)

        hidden_activation = self.neural_network_config.hidden_layer_activation_combo.currentText()
        output_activation = self.neural_network_config.output_layer_activation_combo.currentText()
        activation = NeuralNetworkActivation(hidden_activation, output_activation)

        optimizer = self.neural_network_config.optimizer_combo.currentText()
        loss = self.neural_network_config.loss_combo.currentText()
        loss_and_optimizer = NeuralNetworkLossAndOptimizer(loss, optimizer)

        metrics = self.neural_network_config.metrics_combo.currentText()

        # Create NeuralNetwork model
        self.neural_network = NeuralNetwork(shape, activation, loss_and_optimizer, metrics,
                                            self.parent.preprocessed_data_standard)

        # Build model
        self.built_neural_network_model = self.neural_network.build_model()

    def train_neural_network_model(self):
        stop_neural_network_training_flag.stop = False

        # Build neural network model
        self.build_neural_network()

        # Compile model
        self.neural_network.compile_model()

        try:
            # Measure training time
            start_time = time.time()

            # Train model
            history_reports = self.neural_network.train(int(self.neural_network_config.epochs), int(self.neural_network_config.batch_size))

            end_time = time.time()
            execution_time = end_time - start_time

        except Exception as e:
            self.parent.message_dialog.error(f"Error while training neural network: {e}")
        else:
            # Evaluate model
            eval_reports = self.neural_network.evaluate()

            print(f"Training time: {execution_time}")

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

    def generate_neural_network_plot(self, model, path):
        plot_model(model, to_file=path, show_shapes=True,
                   show_layer_names=True, show_layer_activations=True, show_trainable=True)
        
    
    # Build LSTM network
    def build_and_compile_lstm_network(self):
         # Get LSTM network configuration
        self.lstm_config.batch_size = self.lstm_config.batch_size_edit.text()
        self.lstm_config.epochs = self.lstm_config.epochs_edit.text()
        num_of_units = self.lstm_config.num_of_units_edit.text()
        look_back = self.lstm_config.look_back_edit.text()

        # Check if empty
        if self.lstm_config.batch_size is None or self.lstm_config.epochs is None or num_of_units is None or look_back is None:
            self.parent.message_dialog.error("Fields cannot be empty")
            return

        # Get activation function
        activation = self.lstm_config.activation_function.currentText()

        optimizer = self.lstm_config.optimizer_combo.currentText()
        loss = self.lstm_config.loss_combo.currentText()
        loss_and_optimizer = NeuralNetworkLossAndOptimizer(loss, optimizer)

        metrics = self.lstm_config.metrics_combo.currentText()

        lstm_data_preprocessor = LSTMDataPreprocessor(self.parent.transformed_dataset, data_row, int(look_back))

        # Create and build LSTM model
        self.lstm_network = LSTMNetwork(num_of_units, activation, loss_and_optimizer, metrics,
                                            lstm_data_preprocessor)
        
        # Compile model
        self.lstm_network.compile_model()
        
    def train_lstm_network(self):
        stop_neural_network_training_flag.stop = False

        self.build_and_compile_lstm_network()

        try:
            # Measure training time
            start_time = time.time()

            # Train model
            history_reports = self.lstm_network.train(int(self.lstm_config.epochs), int(self.lstm_config.batch_size))

            end_time = time.time()
            execution_time = end_time - start_time

        except Exception as e:
            self.parent.message_dialog.error(f"Error while training LSTM neural network: {e}")
        else:
            print(f"Training time: {execution_time}")

            # Download reports
            # Prompt report download
            reply = self.parent.message_dialog.question("Download reports",
                                                        "Do you want to download training and evaluation "
                                                        "report?")
            if reply == QMessageBox.Yes:
                # Save reports to csv files
                save_history_and_val_reports_csv_file(self.parent, history_reports)
            else:
                self.parent.message_dialog.information("Training successfully finished!")
