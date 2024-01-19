# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
import sys

from PyQt5.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def main():
    # # Use the dataset as needed
    # dimensions = dataset.dimensions()
    # print("Data dimensions: ", dimensions)
    #
    # first_rows = dataset.head()
    # print("First rows: \n", first_rows)
    #
    # # Plot element vector
    # dataset.plot_element_vector()
    #
    # # Plot histogram
    # dataset.plot_histogram_nk()
    #
    # # Describe rotor angle column
    # dataset.describe_rotor_angle_column()
    #

    #
    # # Show first rows of a transformed dataset
    # first_rows = transformed_dataset.head()
    # print("First rows: \n", first_rows)
    #
    # # Plot correlation matrix
    # dataset.plot_correlation_matrix()

    # Run the application
    app = QApplication(sys.argv)
    # Create a Qt widget, which will be our window.
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()


if __name__ == "__main__":
    main()
