
# Electric Motor Model Identification Application

## Overview
This application provides an interactive GUI for the identification of electric motor models. Utilizing a dataset from [Kaggle](https://www.kaggle.com/datasets/hankelea/system-identification-of-an-electric-motor?resource=download), it enables users to overview the dataset, train linear regression models (with and without hyperparameter tuning), configure neural network models, and display the architecture of these neural networks.

## Features
- **Dataset Overview**: View and analyze the electric motor dataset.
- **Linear Regression Model Training**: Train a linear regression model on the dataset.
  - *With Hyperparameter Tuning*: Optimize the model's performance.
  - *Without Hyperparameter Tuning*: Train the model with default settings.
- **Neural Network Configuration**: Configure and customize neural network models.
- **Neural Network Visualization**: Display the architecture of the neural network models.

## Installation

### Prerequisites
- Python 3.8 or above (See Python installation instructions below)
- pip3 (Python package manager)

### Python Installation
1. Download and install Python from [python.org](https://www.python.org/downloads/).
2. During installation, ensure that you select the option to add Python to your PATH.

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/electric-motor-model-identification.git
   cd electric-motor-model-identification
   ```

2. **Create a Virtual Environment (Optional)**
   ```bash
   python3 -m venv venv
   ```
   - On Windows, activate the virtual environment with:
     ```bash
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS, use:
     ```bash
     source venv/bin/activate
     ```

3. **Install Required Libraries**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Download the Dataset**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hankelea/system-identification-of-an-electric-motor?resource=download).

## Usage

To run the application, first ensure that the virtual environment is activated if you have created one. Then execute:

```bash
python3 main.py
```

## Contributing
Contributions to this project are welcome. Please follow the standard procedure of forking the repository and submitting a pull request.

## License
This project is licensed under the [MIT License](LICENSE.md).
