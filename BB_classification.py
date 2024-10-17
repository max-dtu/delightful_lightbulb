import open3d as o3d
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from o3d_tools.visualize import PointCloudProject
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load point cloud data project-wise
project1 = PointCloudProject(project='Project1')
project2 = PointCloudProject(project='Project2')
project3 = PointCloudProject(project='Project3')
project4 = PointCloudProject(project='Project4')

def stratified_split_with_test(projects, test_size=0.2, random_state=None):
    """
    Splits point cloud data from multiple projects into train+validation and test sets.
    
    Parameters:
    - projects: List of PointCloudProject objects.
    - test_size: Proportion of data for the test set (default is 0.2).
    - random_state: Random seed for reproducibility.
    
    Returns:
    - train_val_df: Combined train+validation dataframe.
    - test_df: Combined test dataframe.
    """
    
    combined_df_list = []

    # Iterate through each project and combine data
    for project in projects:
        objects_df = project.objects_df  # Access the point cloud data for the project
        project_name = project.project
        
        # Append project name to dataframe
        for class_name, df in objects_df.items():
            df['Project'] = project_name
            combined_df_list.append(df)

    # Combine data from all projects
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    # Perform stratified split to get train+validation and test sets
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=test_size,
        random_state=random_state,
        stratify=combined_df[' Label']
    )

    return train_val_df, test_df

# Example usage with four projects
from sklearn.model_selection import cross_val_score
projects = [project1, project2, project3, project4]
train_val_set, test_set = stratified_split_with_test(projects, test_size=0.2, random_state=42)
import numpy as np

# Select features (bounding box coordinates) and the target (label) for training and test sets
X_test = test_set[[' BB.Min.X ', ' BB.Min.Y ', ' BB.Min.Z ', ' BB.Max.X ', ' BB.Max.Y ', ' BB.Max.Z']]
y_test = test_set[' Label']

# Perform cross-validation within the train+validation set
X_train_val = train_val_set[[' BB.Min.X ', ' BB.Min.Y ', ' BB.Min.Z ', ' BB.Max.X ', ' BB.Max.Y ', ' BB.Max.Z']]
y_train_val = train_val_set[' Label']

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation: Use StratifiedKFold for validation within train+validation set
cv = StratifiedKFold(n_splits=5)

# Evaluate using cross_val_score on train+validation data
cross_val_scores = cross_val_score(rf_model, X_train_val, y_train_val, cv=cv, scoring='accuracy')

# Print the cross-validation accuracy scores
for i, score in enumerate(cross_val_scores, 1):
    print(f'Fold {i}: Validation Accuracy = {score:.2f}')

# Train the model on the full train+validation set
rf_model.fit(X_train_val, y_train_val)

# Now evaluate the model on the reserved test set
y_pred_test = rf_model.predict(X_test)

# Assess performance on the test set
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Confusion matrix and classification report for the test set
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_test))

print('Test Classification Report:')
print(classification_report(y_test, y_pred_test))

# Visualize the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate predictions on the test set
y_pred_test = rf_model.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Normalize the confusion matrix to percentages
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6), dpi = 200)
sns.heatmap(conf_matrix, annot=True, fmt="%", cmap="Blues", xticklabels=y_train_val.unique(), yticklabels=y_train_val.unique(), annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize = 12)
plt.xlabel('Predicted Labels', fontsize = 12)
plt.ylabel('True Labels', fontsize = 12)
plt.show()