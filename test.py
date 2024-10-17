import open3d as o3d
import numpy as np


# from o3d_tools.visualize import PointCloudProject
# from extract import PointCloudProject
from util import cal_loss, IOStream
import torch

# project1 = PointCloudProject(project='Project1')

# box_pcd, labels = project1.extract_box_and_label(resample=True, num_points=4096)

# points_np_list = [np.asarray(pcd.points) for pcd in box_pcd]

# points_np = np.array(points_np_list)

# i = 17
# print(labels[i])
# o3d.visualization.draw_geometries([box_pcd[i]])



# Example usage
batch_size = 32
num_points = 10000
num_classes = 2

# Random predictions (logits)
pred = torch.rand(batch_size, num_points, num_classes)

mask = torch.randint(0, 2, (batch_size, num_points))
print(mask)

# Random labels
gold = torch.randint(0, num_classes, (batch_size, num_points))

# Move tensors to the appropriate device (CPU or GPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
pred = pred.to(device)
gold = gold.to(device)

# Reshape pred to (batch_size * num_points, num_classes)
pred = pred.view(-1, num_classes)

# Calculate loss
loss = cal_loss(pred, gold)
print(f'Loss: {loss.item()}')


