import open3d as o3d
import numpy as np
from o3d_tools.io import read_csv, read_points, read_objects, read_masks, read_object_bb, get_global_mask

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
        # Read CSV file
        self.read_csv = read_csv(project)

    def add_mask(self):
        # Read mask
        mask_indices = self.global_mask
        # Create a boolean mask
        mask = np.zeros(np.array(self.pcd.points).shape[0], dtype=bool)
        mask[mask_indices] = True
        # Get only mask subset of point cloud
        masked_points = np.array(self.pcd.points)[mask, :]
        masked_colors = np.array(self.pcd.colors)[mask, :]
        masked_colors = np.array([[0, 1, 0] for row in masked_colors])
        masked_pcd = o3d.geometry.PointCloud()
        masked_pcd.points = o3d.utility.Vector3dVector(masked_points)
        masked_pcd.colors = o3d.utility.Vector3dVector(masked_colors)
        return masked_pcd
    
    def add_invmask(self):
        # Read mask
        mask_indices = self.global_mask
        # Create a boolean mask
        mask = np.zeros(np.array(self.pcd.points).shape[0], dtype=bool)
        mask[mask_indices] = True
        # Get only mask subset of point cloud
        inv_masked_points = np.array(self.pcd.points)[~mask, :]
        inv_masked_colors = np.array(self.pcd.colors)[~mask, :]
        inv_masked_pcd = o3d.geometry.PointCloud()
        inv_masked_pcd.points = o3d.utility.Vector3dVector(inv_masked_points)
        inv_masked_pcd.colors = o3d.utility.Vector3dVector(inv_masked_colors)
        return inv_masked_pcd

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

    def resample_points_and_points_labels(self, pcd, points_labels, num_points):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        points_labels = np.asarray(points_labels)
        num_rows = len(points)

        if num_rows > num_points:
            # If there are more points, randomly select a subset
            indices = np.random.choice(num_rows, num_points, replace=False)
            resampled_points = points[indices]
            resampled_colors = colors[indices]
            resampled_points_labels = points_labels[indices]
        elif num_rows < num_points:
            # If there are fewer points, randomly sample points to fill
            diff = num_points - num_rows
            sampled_indices = np.random.choice(num_rows, diff, replace=True)
            sampled_points = points[sampled_indices]
            sampled_colors = colors[sampled_indices]
            sampled_points_labels = points_labels[sampled_indices]
            resampled_points = np.vstack((points, sampled_points))
            resampled_colors = np.vstack((colors, sampled_colors))
            resampled_points_labels = np.hstack((points_labels, sampled_points_labels))
        else:
            # If exactly `num_points`, keep the array as is
            resampled_points = points
            resampled_colors = colors
            resampled_points_labels = points_labels

        # Create a new point cloud object with the resampled points and colors
        resampled_pcd = o3d.geometry.PointCloud()
        resampled_pcd.points = o3d.utility.Vector3dVector(resampled_points)
        resampled_pcd.colors = o3d.utility.Vector3dVector(resampled_colors)
        
        return resampled_pcd, resampled_points_labels

    def extract_box(self, resample=False, num_points=1024):
        boxed_pcd = []

        for label in self.objects.keys():
            for bb in self.objects[label]:
                cropped_pcd = self.pcd.crop(bb)
                if resample:
                    cropped_pcd = self.resample_points(cropped_pcd, num_points)
                boxed_pcd.append(cropped_pcd)

        return boxed_pcd
    
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
    
    def extract_box_and_points_labels(self, resample=False, num_points=1024):
        points_labels = []
        boxed_pcd = []

        for label in self.objects.keys():
            for bb in self.objects[label]:
                cropped_mask_pcd = self.add_mask().crop(bb)
                cropped_inv_mask_pcd = self.add_invmask().crop(bb)

                # mask_labels = np.vstack((np.ones(np.array(cropped_mask_pcd.points).shape[0]), np.zeros(np.array(cropped_mask_pcd.points).shape[0]))).T
                # inv_mask_labels = np.vstack((np.zeros(np.array(cropped_inv_mask_pcd.points).shape[0]), np.ones(np.array(cropped_inv_mask_pcd.points).shape[0]))).T

                mask_labels = np.ones(np.array(cropped_mask_pcd.points).shape[0])
                inv_mask_labels = np.zeros(np.array(cropped_inv_mask_pcd.points).shape[0])


                mask_points = np.array(cropped_mask_pcd.points)
                inv_mask_points = np.array(cropped_inv_mask_pcd.points)

                mask_colors = np.array(cropped_mask_pcd.colors)
                inv_mask_colors = np.array(cropped_inv_mask_pcd.colors)

                cropped_pcd_labels = np.concatenate((mask_labels, inv_mask_labels))

                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(np.concatenate((mask_points, inv_mask_points)))      
                cropped_pcd.colors = o3d.utility.Vector3dVector(np.concatenate((mask_colors, inv_mask_colors)))  

                # permutation = np.random.permutation(len(cropped_pcd.points))
                # # Shuffle both arrays using the permutation
                # cropped_pcd.points = o3d.utility.Vector3dVector(np.array(cropped_pcd.points)[permutation])
                # cropped_pcd.colors = o3d.utility.Vector3dVector(np.array(cropped_pcd.colors)[permutation])
                # cropped_pcd_labels = cropped_pcd_labels[permutation]

                if resample:
                    cropped_pcd, cropped_pcd_labels = self.resample_points_and_points_labels(cropped_pcd, cropped_pcd_labels, num_points)
                
                points_labels.append(cropped_pcd_labels)
                boxed_pcd.append(cropped_pcd)

        return boxed_pcd, points_labels