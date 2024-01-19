# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog


class InitialDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Electric Motor Identification")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        message_label = QLabel("Welcome to the Electric Motor Identification App! "
                               "To proceed, please upload dataset in CSV format.")
        layout.addWidget(message_label)

        upload_button = QPushButton("Upload")
        upload_button.clicked.connect(self.upload_csv)
        layout.addWidget(upload_button)

        self.setLayout(layout)

    def upload_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        if file_name:
            print(f"Selected CSV File: {file_name}")