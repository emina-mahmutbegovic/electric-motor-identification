# Copyright (c) 2024 Emina Mahmutbegovic
#
# All rights reserved.
# This software is the proprietary information of Emina Mahmutbegovic
# Unauthorized sharing of this file is strictly prohibited
from dataset.dataset import Dataset
from linear_regression.linear_regression import LinearRegressionModel
from util.util import generate_mse_report
from util.util import generate_report
from neural_network.neural_network import NeuralNetworkShape
from neural_network.neural_network import NeuralNetwork


def main():
    # Create an instance of the Dataset class
    dataset = Dataset()

    # Use the dataset as needed
    dimensions = dataset.dimensions()
    print("Data dimensions: ", dimensions)

    first_rows = dataset.head()
    print("First rows: \n", first_rows)

    # Plot element vector
    dataset.plot_element_vector()

    # Plot histogram
    dataset.plot_histogram_nk()

    # Describe rotor angle column
    dataset.describe_rotor_angle_column()

    # Transform data
    transformed_dataset = dataset.transform()

    # Show first rows of a transformed dataset
    first_rows = transformed_dataset.head()
    print("First rows: \n", first_rows)

    # Plot correlation matrix
    dataset.plot_correlation_matrix()

    # # Create LinearRegression model
    # linear_regression = LinearRegressionModel(transformed_dataset)
    #
    # # Train model and predict scores
    # scores = linear_regression.train_and_predict()
    #
    # # Print scores
    # linear_regression.print_scores(scores)
    #
    # # Train model by using hyperparameter tuning and predict scores
    # scores = linear_regression.hyper_parameter_tuning()
    #
    # # Print scores
    # linear_regression.print_scores(scores)
    #
    # # Generate report
    # generate_report(scores)

    # Create NeuralNetwork model
    neural_network_shape = NeuralNetworkShape((19,), [2, 1], 2)
    neural_network = NeuralNetwork(neural_network_shape, transformed_dataset)

    # Build model
    neural_network.build_model()

    # Compile model
    neural_network.compile_model()

    # Train model
    history_reports = neural_network.train(1, 16)

    # Generate report
    generate_report(history_reports, 'Loss and accuracy for training data:')

    # Evaluate model
    eval_reports = neural_network.evaluate()

    # Generate report
    generate_report(str(eval_reports), 'Loss and accuracy for evaluation data:')


if __name__ == "__main__":
    main()
