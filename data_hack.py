import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_project_data(project_name, label_encoder, num_points):
    csv_file = os.path.join('data', 'TrainingSet', project_name, f'{project_name}.csv')
    xyz_file = os.path.join('data', 'TrainingSet', project_name, f'{project_name}.xyz')
    
    # Load labels and bounding boxes from the CSV file
    df = pd.read_csv(csv_file)
    labels = df.iloc[:, 1].values  # Assuming the second column is the label
    bounding_boxes = df.iloc[:, 2:].values  # The remaining columns are the bounding box coordinates
    
    # Load point cloud data from the XYZ file
    try:
        points = np.loadtxt(xyz_file)
    except ValueError as e:
        print(f"Error loading {xyz_file}: {e}")
        return np.array([]), np.array([])

    # Encode labels
    labels = label_encoder.transform(labels)
    
    all_points = []
    all_labels = []
    
    for label, bbox in zip(labels, bounding_boxes):
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        # Extract points within the bounding box
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        bbox_points = points[mask]
        
        # If there are more points than num_points, randomly sample num_points points
        if bbox_points.shape[0] > num_points:
            indices = np.random.choice(bbox_points.shape[0], num_points, replace=False)
            bbox_points = bbox_points[indices]
        # If there are fewer points than num_points, pad with zeros
        elif bbox_points.shape[0] < num_points:
            padding = np.zeros((num_points - bbox_points.shape[0], bbox_points.shape[1]))
            bbox_points = np.vstack((bbox_points, padding))
        
        # Ensure the points have 3 features (x, y, z)
        bbox_points = bbox_points[:, :3]
        
        all_points.append(bbox_points)
        all_labels.append(label)
    
    if all_points:
        all_points = np.array(all_points)
        all_labels = np.array(all_labels)
    else:
        all_points = np.array([])
        all_labels = np.array([])
    
    return all_points, all_labels

def load_data(test_size=0.2, num_points=1024):
    all_data = []
    all_labels = []

    # Initialize label encoder with all possible labels
    label_encoder = LabelEncoder()
    all_possible_labels = ['Pipe', 'HVAC_Duct', 'Structural_IBeam', 'Structural_ColumnBeam']
    label_encoder.fit(all_possible_labels)

    for project_name in ['Project1', 'Project2', 'Project3', 'Project4']:
        points, labels = load_project_data(project_name, label_encoder, num_points)
        if points.size > 0 and labels.size > 0:
            all_data.append(points)
            all_labels.append(labels)

    # Flatten the lists of arrays
    if all_data:
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_data = np.array([])
        all_labels = np.array([])

    # Verify that all arrays have the same length
    assert len(all_data) == len(all_labels), "Inconsistent data lengths"

    # Split the data into training and testing sets
    if all_data.size > 0 and all_labels.size > 0:
        train_data, test_data, train_labels, test_labels = train_test_split(
            all_data, all_labels, test_size=test_size, random_state=42
        )
    else:
        train_data, test_data, train_labels, test_labels = np.array([]), np.array([]), np.array([]), np.array([])

    return train_data, test_data, train_labels, test_labels