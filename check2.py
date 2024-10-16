import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_hack import load_data  # Import the data loading function

def plot_point_cloud(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def main(args):
    # Load data using the data loading function
    train_data, test_data, train_labels, test_labels = load_data(num_points=args.num_points)

    # Print the shape of the data to determine num_features
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Plot a sample of the point cloud data with labels
    # Ensure that the labels match the number of points
    for i in range(len(train_data)):
        if train_labels[i] == 0:
            sample_data = train_data[i]
            sample_labels = np.full(sample_data.shape[0], train_labels[i])
            plot_point_cloud(sample_data, sample_labels)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Point Cloud Data')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to use')
    args = parser.parse_args()

    main(args)