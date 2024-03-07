# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import sys
import signal

from PyQt5.QtWidgets import QApplication

from src.ui.main_window import MainWindow

from src.util.shared import stop_training_flag


# Signal handler function
def signal_handler(sig, frame):
    print('\nSIGINT (Ctrl+C) received. Setting a flag to stop training at the end of this epoch.')
    stop_training_flag.stop = True


def main():
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Run the application
    app = QApplication(sys.argv)
    # Create a Qt widget, which will be our window.
    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()


if __name__ == "__main__":
    main()
