# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import sys

from PyQt5.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def main():
    # Run the application
    app = QApplication(sys.argv)
    # Create a Qt widget, which will be our window.
    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()


if __name__ == "__main__":
    main()
