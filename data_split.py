#%%
import open3d as o3d
import pandas as pd
from sklearn.model_selection import train_test_split
from o3d_tools.visualize import PointCloudProject
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load point cloud data project-wise
project1 = PointCloudProject(project='Project1')
project2 = PointCloudProject(project='Project2')
project3 = PointCloudProject(project='Project3')
project4 = PointCloudProject(project='Project4')

#%% Plot distributions of bounding box dimensions for each class
from sklearn.preprocessing import StandardScaler

projects = [project1, project2, project3, project4]

# Iterate through each project
dfs_list = []

for project in projects:
    # Get the point cloud data for each project
    objects_df = project.objects_df  # Adjust for how data is accessed
    project_name = project.project
    
    # Split class-specific data
    for class_name, df in objects_df.items():
        # Add project name as a column
        df['Project'] = project_name

        dfs_list.append(df)

combined_dfs = pd.concat(dfs_list, ignore_index=True)
type_dict = {'HVAC_duct': 'Duct', 'Pipe': 'Pipe', 'Structural_ColumnBeam': 'Column Beam', 'Structural_IBeam': 'IBeam'}

combined_dfs[' Label'] = combined_dfs[' Label'].map(type_dict)

# Compute height, width, and depth for each object in the combined dataset
combined_dfs['Height'] = combined_dfs[' BB.Max.Y '] - combined_dfs[' BB.Min.Y ']
combined_dfs['Width'] = combined_dfs[' BB.Max.X '] - combined_dfs[' BB.Min.X ']
combined_dfs['Depth'] = combined_dfs[' BB.Max.Z'] - combined_dfs[' BB.Min.Z ']

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the figure size and create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi = 200)

# List of features to plot
features = ['Height', 'Width', 'Depth']

# Define a color palette for the labels
palette = sns.color_palette('Set2')

# Iterate over the features and create a distribution plot for each one
for i, feature in enumerate(features):
    sns.boxplot(data=combined_dfs, x=' Label', y=feature, ax=axes[i], palette=palette)
    # sns.kdeplot(data=combined_dfs, x=feature, hue=' Label', ax=axes[i], fill=True, common_norm=False, palette=palette)
    axes[i].set_title(f'Distribution of {feature}', fontsize = 16)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='both', which='major', labelsize=16)  
    axes[i].grid(True)

# Add a global title to the figure
plt.suptitle('Distribution of Bounding Box Dimensions by Object', fontsize=16)

# Display the plot
plt.tight_layout()
plt.savefig('bounding_box_dimensions.png')
plt.show()

# # Standardize the height, width, and depth
# scaler = StandardScaler()
# combined_dfs[['Height', 'Width', 'Depth']] = scaler.fit_transform(combined_dfs[['Height', 'Width', 'Depth']])

# # Set the figure size and create subplots
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # List of features to plot
# features = ['Height', 'Width', 'Depth']

# # Define a color palette for the labels
# palette = sns.color_palette('Set2')

# # Iterate over the features and create a boxplot for each one
# for i, feature in enumerate(features):
#     # sns.kdeplot(data=combined_dfs, x=feature, hue=' Label', ax=axes[i], fill=True, common_norm=False, palette=palette)
#     sns.boxplot(data=combined_dfs, x=' Label', y=feature, ax=axes[i], palette=palette)
#     axes[i].set_title(f'Standardized {feature} by Label')
#     axes[i].set_xlabel('Label')
#     axes[i].set_ylabel(f'Standardized {feature}')

# # Add a global title to the figure
# plt.suptitle('Standardized Boxplots of Bounding Box Dimensions (Height, Width, Depth) by Label', fontsize=16)

# # Display the plot
# plt.tight_layout()
# plt.show()

#%% Randon Forest Classifier
def stratified_train_test_split(projects, test_size=0.2, random_state=None):
    """
    Splits point cloud data from multiple projects into stratified train and test sets.
    
    Parameters:
    - projects: List of PointCloudProject objects.
    - test_size: Proportion of data for the test set (default is 0.2).
    - random_state: Random seed for reproducibility.
    
    Returns:
    - train_df_combined: Combined train dataframe.
    - test_df_combined: Combined test dataframe.
    """
    
    train_dfs_list = []
    test_dfs_list = []

    # Iterate through each project
    for project in projects:
        # Get the point cloud data for each project
        objects_df = project.objects_df  # Adjust for how data is accessed
        project_name = project.project
        
        # Split class-specific data
        for class_name, df in objects_df.items():
            # Add project name as a column
            df['Project'] = project_name

            # Perform stratified split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[' Label']
            )
            train_dfs_list.append(train_df)
            test_dfs_list.append(test_df)

    # Combine train and test data from all projects
    train_df_combined = pd.concat(train_dfs_list, ignore_index=True)
    test_df_combined = pd.concat(test_dfs_list, ignore_index=True)

    return train_df_combined, test_df_combined

#%% Example usage with four projects
from sklearn.preprocessing import StandardScaler, LabelEncoder

projects = [project1, project2, project3, project4]
train_set, test_set = stratified_train_test_split(projects, test_size=0.2, random_state=42)

# Encode labels as categories
label_encoder = LabelEncoder()

# Select features (bounding box coordinates) and the target (label) for training and testing
X_train = train_set[[' BB.Min.X ', ' BB.Min.Y ', ' BB.Min.Z ', ' BB.Max.X ', ' BB.Max.Y ', ' BB.Max.Z']]
y_train = label_encoder.fit_transform(train_set[' Label'])
X_test = test_set[[' BB.Min.X ', ' BB.Min.Y ', ' BB.Min.Z ', ' BB.Max.X ', ' BB.Max.Y ', ' BB.Max.Z']]
y_test = label_encoder.transform(test_set[' Label'])

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Assess the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Detailed classification report (precision, recall, F1-score)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# %% Nested Cross-Validation
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Outer cross-validation loop (for model performance evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner cross-validation loop for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV for inner cross-validation loop
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=inner_cv, scoring='accuracy')

# Nested cross-validation loop for overall performance estimation
nested_cv_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring='accuracy')

# Print the accuracy for each outer fold
for i, score in enumerate(nested_cv_scores, 1):
    print(f'Outer Fold {i}: Accuracy = {score:.2f}')

# Print mean and standard deviation of the nested cross-validation scores
print(f'Nested CV Accuracy: {nested_cv_scores.mean():.2f} ± {nested_cv_scores.std():.2f}')

# Now, train the final model using the best hyperparameters from GridSearchCV
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Assess the performance on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Set Accuracy: {accuracy:.2f}')

#%% Print the best hyperparameters
print("Best Hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)

#Detailed classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Normalize the confusion matrix to show percentages
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Create a custom annotation format with the percentage sign
labels = np.array([["{:.1f}%".format(value) for value in row] for row in conf_matrix_normalized])

type_dict = {'HVAC_duct': 'Duct', 'Pipe': 'Pipe', 'Structural_ColumnBeam': 'Column Beam', 'Structural_IBeam': 'IBeam'}

plt.figure(figsize=(6, 6), dpi = 200)
sns.heatmap(conf_matrix_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=type_dict.values(),
            yticklabels=type_dict.values(),
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize = 14)
plt.xlabel('Predicted Labels', fontsize = 14)
plt.ylabel('True Labels', fontsize = 14)
plt.show()

# %% Retrain the model on the entire dataset and run inference
# Combine the training and test sets
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# Train the final model on the entire dataset
final_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)

final_model.fit(X, y)

# Load X_test and y_test from the previous code cell
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Assess the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Detailed classification report (precision, recall, F1-score)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
