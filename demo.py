import open3d as o3d
from o3d_tools.visualize import PointCloudProject

# Load point cloud data project-wise
project1 = PointCloudProject(project='Project1')
# project2 = PointCloudProject(project='Project2')
# project3 = PointCloudProject(project='Project3')
# project4 = PointCloudProject(project='Project4')


# Draw point cloud and add Pipe bounding boxes
o3d.visualization.draw_geometries([project1.pcd] + list(project1.objects['Pipe']))

# Draw point cloud and add bounding boxes for all three object types
o3d.visualization.draw_geometries([project1.pcd]
                                  + list(project1.objects['Pipe'])
                                  + list(project1.objects['Structural_IBeam'])
                                  + list(project1.objects['Structural_ColumnBeam'])
                                  )


# choose which object types to draw/crop from: 'Pipe', 'HVAC_Duct', 'Structural_ColumnBeam' or 'Structural_IBeam'
which = ['Pipe', 'Structural_IBeam']
_ = project1.draw_bb_only(which, plot=True)
_ = project1.draw_bb_only(which, masked=True, plot=True)

# Draw only subset of project2 as defined by Pipe and Structural_IBeam bounding boxes
o3d.visualization.draw_geometries(project1.draw_bb_only(which, plot=False))

# Draw entire point cloud with mask
o3d.visualization.draw_geometries([project1.add_mask()])

# Draw entire point cloud, but removing all points belonging to bounding boxes of object types as defined below
# This is very slow! Uncomment if execution is desirable!
pcd_without_objects = project1.draw_bb_inverse(which_remove=['Pipe', 'Structural_IBeam', 'Structural_ColumnBeam'])
