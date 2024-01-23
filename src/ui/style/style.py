# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from PIL.ImageQt import rgb

from src.util.util import get_file_path

# Def colors
green_theme_color = "rgb(133, 230, 191)"


# Set background image
def set_background_image(parent):
    background_image_file_path = get_file_path('src/ui/assets/background.png')

    parent.setStyleSheet(f"QMainWindow {{background-image: url({background_image_file_path}); "
                         f"background-repeat: no-repeat; background-position: center;}}")


# Set label stylesheet
def generate_label_stylesheet(font_size, color="black"):
    return f"QLabel {{color: {color}; font-weight: {font_size};}}"


# Button style
def generate_button_stylesheet():
    return f"QPushButton {{ color: {green_theme_color}; font-weight: bold;}}"


# Combo box style
def generate_combobox_stylesheet():
    return f"QComboBox {{ color: {green_theme_color}; font-weight: bold;}}"


# Line edit style
def generate_line_edit_stylesheet():
    return f"QLineEdit {{ color: {green_theme_color}; font-weight: bold;}}"
