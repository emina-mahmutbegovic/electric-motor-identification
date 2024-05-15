# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTabWidget, QLabel, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt

from src.ui.style.style import generate_label_stylesheet


# DataFrame model for transformation of dataset.head to table model
class DataFrameModel(QAbstractTableModel):
    def __init__(self, data):
        super(DataFrameModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._data.columns[section]
        else:
            return str(self._data.index[section])


# Main DatasetView class
class DatasetView:
    def __init__(self, parent):
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        # Define bold stylesheet
        bold_label_style = generate_label_stylesheet("bold")

        # Remove background image
        self.parent.setStyleSheet("QMainWindow {background-image: none;}")

        tab_widget = QTabWidget()
        self.parent.setCentralWidget(tab_widget)

        # Tab 1: Overall Information
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1.setLayout(tab1_layout)
        tab1_label = QLabel(f'Dataset dimensions: {self.parent.dataset.dimensions()}')
        tab1_label.setStyleSheet(bold_label_style)
        tab1_layout.addWidget(tab1_label)

        # Initialize DataFrame model
        model = DataFrameModel(self.parent.dataset.head())
        table_view = QTableView()
        table_view.setModel(model)
        # Add heading
        tab1_heading = QLabel('Dataset sample:')
        tab1_heading.setStyleSheet(bold_label_style)
        tab1_layout.addWidget(tab1_heading)
        # Set DataFrame model
        tab1_layout.addWidget(table_view)

        # Tab 2: Element Vector Plot
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        tab2.setLayout(tab2_layout)
        tab2_layout.addWidget(self.parent.dataset.plot_element_vector())

        # Tab 3: Histogram
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3.setLayout(tab3_layout)
        tab3_layout.addWidget(self.parent.dataset.plot_histogram_nk())

        # Tab 5: Transformed Dataset
        tab4 = QWidget()
        tab4_layout = QVBoxLayout()
        tab4.setLayout(tab4_layout)
        # Initialize transformed_dataset model
        transformed_model = DataFrameModel(self.parent.transformed_dataset.head())
        transformed_model_table_view = QTableView()
        transformed_model_table_view.setModel(transformed_model)

        # Set heading
        tab5_heading = QLabel('Transformed dataset sample:')
        tab5_heading.setStyleSheet(bold_label_style)
        tab4_layout.addWidget(tab5_heading)
        tab4_layout.addWidget(transformed_model_table_view)

        # Tab 6: Pearson Coefficient
        tab5 = QWidget()
        tab5_layout = QVBoxLayout()
        tab5.setLayout(tab5_layout)
        tab5_layout.addWidget(self.parent.dataset.plot_correlation_map(self.parent.pearson_coefficients))

        # Tab 6: Spearman Coefficient
        tab6 = QWidget()
        tab6_layout = QVBoxLayout()
        tab6.setLayout(tab6_layout)
        tab6_layout.addWidget(self.parent.dataset.plot_correlation_map(self.parent.spearman_coefficients))

        tab_widget.addTab(tab1, "Overall Information")
        tab_widget.addTab(tab2, "Element Vector Plot")
        tab_widget.addTab(tab3, "Histogram")
        tab_widget.addTab(tab4, "Transformed Dataset")
        tab_widget.addTab(tab5, "Pearson Coefficients")
        tab_widget.addTab(tab6, "Spearman Coefficients")


