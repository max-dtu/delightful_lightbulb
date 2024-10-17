#%% Import and preprocess pcd data
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

#%% Define model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# PointNet Model Definition
class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        
        # Shared MLP (point-wise feature extraction)
        self.conv1 = nn.Conv1d(3, 64, 1)   # 3 input dims (x, y, z), 64 output dims
        self.conv2 = nn.Conv1d(64, 128, 1) # 64 input dims, 128 output dims
        self.conv3 = nn.Conv1d(128, 1024, 1) # 128 input dims, 1024 output dims
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Input x has shape (batch_size, num_points, 3), permute to (batch_size, 3, num_points)
        x = x.permute(0, 2, 1)

        # Apply shared MLP
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=False)[0]

        # Fully connected layers
        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here as it's handled in the loss function

        return x
    
#%% Train model 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# Split dataset into training and validation sets
X = resize_and_concatenate(boxed_pcds, 1024)
y = np.array(all_labels)

# Encode labels as categories
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Stratified split into training, validation, and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_index, test_index = next(sss.split(X, y))
X_train_val, X_test = X[train_val_index], X[test_index]
y_train_val, y_test = y[train_val_index], y[test_index]

# Further split training and validation sets
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 of the total data
train_index, val_index = next(sss_val.split(X_train_val, y_train_val))
X_train, X_val = X_train_val[train_index], X_train_val[val_index]
y_train, y_val = y_train_val[train_index], y_train_val[val_index]

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Move data to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Define the model, optimizer, and loss function
num_classes = 4  # Adjust based on your dataset
model = PointNet(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    
    # Split data into batches
    for i in range(0, X_train.size(0), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation at each epoch
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, predicted = torch.max(val_outputs, 1)
        val_accuracy = (predicted == y_val).float().mean()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy.item():.4f}')

print('Training completed!')
# %% Inference
# Assuming `X_test` is your new data for inference, shape (batch_size, 1024, 3)
# Ensure the model is in evaluation mode
from sklearn.metrics import f1_score, confusion_matrix

def compute_metrics(y_true, y_pred, num_classes):
    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # F1 Score (macro average)
    f1 = f1_score(y_true, y_pred, average='macro')

    # IoU (Intersection over Union)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    intersection = cm.diagonal()  # True Positives
    ground_truth_set = cm.sum(axis=1)  # Actual instances
    predicted_set = cm.sum(axis=0)  # Predicted instances
    
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union  # IoU for each class
    iou = iou[union > 0]  # Ignore classes with no predictions
    mean_iou = iou.mean() if iou.size > 0 else 0  # Mean IoU

    return accuracy, f1, mean_iou

# Example usage after inference
model.eval()

# Assuming `y_val` are the true labels and `X_val` is the validation dataset
with torch.no_grad():
    val_outputs = model(X_val)
    _, predicted = torch.max(val_outputs, 1)

# Convert to numpy
y_val = y_val.cpu().numpy()
predicted = predicted.cpu().numpy()

# Compute metrics
num_classes = 4  # Adjust based on your dataset
accuracy, f1, mean_iou = compute_metrics(y_val, predicted, num_classes)

print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Mean IoU: {mean_iou:.4f}')

# Assuming `y_val` are the true labels and `X_val` is the validation dataset
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)

# Convert to numpy
y_test = y_test.cpu().numpy()
predicted = predicted.cpu().numpy()

accuracy, f1, mean_iou = compute_metrics(y_test, predicted, num_classes)

print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Mean IoU: {mean_iou:.4f}')


# %%
