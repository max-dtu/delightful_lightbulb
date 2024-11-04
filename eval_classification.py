
import open3d as o3d
import os

import torch
import numpy as np
import pandas as pd

# Define the model architecture (assuming DGCNN_semseg_s3dis is defined in model.py)
from model import DGCNN
from o3d_tools.io import read_csv, read_points

# Path to the training set folders
training_set_path = 'data/HiddenSet'

# Find all folder names in the training set directory
project_names = [f for f in os.listdir(training_set_path) if os.path.isdir(os.path.join(training_set_path, f))]
project_names.sort()

print("Project names:", project_names)

all_points_np_list = []
all_ids_np_list = []

# Set environment variables for debugging and multiprocessing
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 

# Determine the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model_path = "outputs_classification/exp/model_final_v1.pth"
num_points = 1024


args = type('', (), {})()  # Create an empty args object
args.k = 20
args.emb_dims = 1024
args.dropout = 0.5
model = DGCNN(args).to(device)

state_dict = torch.load(model_path, map_location=device)
    
# Remove 'module.' prefix if present
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

int_to_name = {0: 'HVAC_Duct', 1: 'Pipe', 2: 'Structural_ColumnBeam', 3: 'Structural_IBeam'}

# Iterate over each project name
for project_name in project_names:
    # project = PointCloudProject(project=project_name)
    # box_pcd = project.extract_box(resample=True, num_points=2048)
    pcd = read_points(project_name)
    objects = pd.read_csv("data/{}/{}/{}.csv".format("HiddenSet", project_name, project_name),
                          sep=",", header=0)
    
    # bbs = objects.iloc[:, 2:8]
    
    for index, row in objects.iterrows():
        min_bound = np.array(row.iloc[2:5])
        max_bound = np.array(row.iloc[5:])
        bb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cropped_pcd = pcd.crop(bb)
        points = np.asarray(pcd.points)
        num_rows = len(points)
        indices = np.random.choice(num_rows, num_points, replace=False)
        points = torch.tensor(points[indices], dtype=torch.float32).to(device)
        points = points.unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            data = points.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1]
            preds_np = preds.cpu().numpy()
            # print("ID:", row.iloc[0], "Prediction:", int_to_name[preds_np[0]])
            print(int_to_name[preds_np[0]])