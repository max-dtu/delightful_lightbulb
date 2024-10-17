import open3d as o3d
import numpy as np
from o3d_tools.io import read_points, read_objects, read_masks, read_object_bb, get_global_mask


class PointCloudProject:
    def __init__(self, project):
        self.project = project
        # Read Point cloud file
        self.pcd = read_points(project)
        # Read objects bounding boxes as dict of pandas df's
        self.objects_df = read_objects(project)
        # Convert object bounding boxes into Open3d objects (for drawing purposes)
        self.objects = read_object_bb(self.objects_df)
        # Read object mask as dict
        self.masks = read_masks(project)
        # Convert to global mask (for drawing purposes)
        self.global_mask = get_global_mask(self.masks)


    def resample_points(self, pcd, num_points):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        num_rows = len(points)

        if num_rows > num_points:
            # If there are more points, randomly select a subset
            indices = np.random.choice(num_rows, num_points, replace=False)
            resampled_points = points[indices]
            resampled_colors = colors[indices]
        elif num_rows < num_points:
            # If there are fewer points, randomly sample points to fill
            diff = num_points - num_rows
            sampled_indices = np.random.choice(num_rows, diff, replace=True)
            sampled_points = points[sampled_indices]
            sampled_colors = colors[sampled_indices]
            resampled_points = np.vstack((points, sampled_points))
            resampled_colors = np.vstack((colors, sampled_colors))
        else:
            # If exactly `num_points`, keep the array as is
            resampled_points = points
            resampled_colors = colors

        # Create a new point cloud object with the resampled points and colors
        resampled_pcd = o3d.geometry.PointCloud()
        resampled_pcd.points = o3d.utility.Vector3dVector(resampled_points)
        resampled_pcd.colors = o3d.utility.Vector3dVector(resampled_colors)
        
        return resampled_pcd

    def extract_box_and_label(self, resample=False, num_points=1024):
        labels = []
        boxed_pcd = []

        for label in self.objects.keys():
            for bb in self.objects[label]:
                cropped_pcd = self.pcd.crop(bb)
                if resample:
                    cropped_pcd = self.resample_points(cropped_pcd, num_points)
                labels.append(label)
                boxed_pcd.append(cropped_pcd)

        return boxed_pcd, labels