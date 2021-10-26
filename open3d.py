import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys

# only needed for tutorial, monkey patches visualization
sys.path.append('..')
import open3d_tutorial as o3dtut
# change to True if you want to interact with the visualization windows
o3dtut.interactive = not "CI" in os.environ

import os, os.path
DIR = '/home/songming/velodyne_points/data'
num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print (num_iter)

vis = o3d.visualization.Visualizer()
vis.create_window()


for i in range(num_iter):
    print("/home/songming/velodyne_points/data/%010d.bin"%i)
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    source = o3d_pcd.voxel_down_sample(voxel_size=0.2)
    print(source)
    
    if i == 0:
        vis.add_geometry(source)
        print("initialization finished")
    else:
        vis.clear_geometries()
        vis.add_geometry(source)
        
        print("update finished")
        
    vis.poll_events()
    vis.update_renderer()
    
vis.destroy_window()
