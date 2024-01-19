# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import QMessageBox


class MessageDialog:
    def __init__(self, parent):
        self.parent = parent

    def information(self, message):
        QMessageBox.information(self.parent, "Information", message)

    def warning(self, message):
        QMessageBox.warning(self.parent, "Warning", message)

    def question(self, title, message):
        return QMessageBox.question(self.parent, title, message,
                                    QMessageBox.Yes | QMessageBox.No)

    def error(self, message):
        QMessageBox.critical(self.parent, "Error", message)
