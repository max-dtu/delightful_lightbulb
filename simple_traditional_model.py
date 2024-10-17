import open3d as o3d
from o3d_tools.data_loader import PointCloudProject
import numpy as np

# Load point cloud data project-wise
project1 = PointCloudProject(project='Project1')
project2 = PointCloudProject(project='Project2')
project3 = PointCloudProject(project='Project3')
project4 = PointCloudProject(project='Project4')

boxed_pcd, labels = project1.extract_box_and_label()

def resize_and_concatenate(arrays, num_points):
    resized_arrays = []

    # Loop through each array
    for arr in arrays:
        num_rows, num_cols = arr.shape

        if num_rows > num_points:
            # If there are more points, randomly sample `num_points`
            indices = np.random.choice(num_rows, num_points, replace=False)
            resized_arr = arr[indices]
        elif num_rows < num_points:
            # If there are fewer points, randomly sample points to fill
            diff = num_points - num_rows
            sampled_indices = np.random.choice(num_rows, diff, replace=True)
            sampled_rows = arr[sampled_indices]
            resized_arr = np.vstack((arr, sampled_rows))
        else:
            # If exactly `num_points`, keep the array as is
            resized_arr = arr

        # Append the resized array to the list
        resized_arrays.append(resized_arr)

    # Concatenate all the resized arrays along axis 0
    # concatenated_array = np.vstack(resized_arrays)

    return np.array(resized_arrays)

boxed_pcds = []
all_labels = []

for project in [project1, project2, project3, project4]:
    boxed_pcd, labels = project.extract_box_and_label()
    boxed_pcds = boxed_pcds + boxed_pcd
    all_labels = all_labels + labels

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you have data in arrays (X) and corresponding labels (y)
# X should be of shape (n_objects, 3) and y of shape (n_objects,)
# For this example, let's assume X and y are loaded or created beforehand.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.reshape(resize_and_concatenate(boxed_pcds, 1024), (450, 3072)), all_labels, test_size=0.2, random_state=42)

# Example 1: Using SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on test data
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")

# Example 2: Using Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

