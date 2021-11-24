import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os, os.path



DIR = '/home/songming/velodyne_points/data'
num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print (num_iter)


def remove_ground(o3d_pcd):
    ##The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. 
    ##It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
    
    
    points = np.asarray(o3d_pcd.points)
    
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=10))
    
    normals = np.asarray(o3d_pcd.normals)
   
    angular_distance_to_z = np.abs(normals[:, 2])
    
    # angles range along the vertical axis (0, pi/6) normal direction constraint
    
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/5) 
    
    idx_points = points[:, 2] < -1.4
    
    idx_2 = np.logical_and(idx_downsampled, idx_points)
    
    idx_2_not = np.logical_not(idx_2)
    
    ground_points = points[idx_2]
    
    up_points = points[idx_2_not] 
    
    
    o3d_ground_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
    
    o3d_up_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(up_points))
    
    plane_model, inliers = o3d_ground_cloud.segment_plane(distance_threshold=0.2,
                                      ransac_n=3,
                                      num_iterations=30)
    # [a, b, c, d] = plane_model
    # print(f" Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    ground_cloud = o3d_ground_cloud.select_by_index(inliers)
    ground_cloud.paint_uniform_color([1.0, 0, 0])
    
    ground_outlier_cloud =  o3d_ground_cloud.select_by_index(inliers, True)
    
    # ground_outlier.paint_uniform_color([0,0,1])
    
    # o3d_up_points.paint_uniform_color([0,0,1])
    
    ground_outlier_points = np.asarray(ground_outlier_cloud.points)
    
    ground_free_points = np.concatenate((ground_outlier_points,up_points), axis=0)
    
    ground_free_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_free_points))
    
    return ground_free_cloud
    
    # ground_free_cloud.paint_uniform_color([0,0,1])
    
    
# vis = o3d.visualization.Visualizer()
# vis.create_window()
num_iter = 10

for i in range(num_iter):
    
    print("/home/songming/velodyne_points/data/%010d.bin"%i)
    
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    
    points = bin_pcd.reshape((-1, 4))[:, 0:3] 
    
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    ##The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. 
    ##It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
    
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=10))
    
    normals = np.asarray(o3d_pcd.normals)
   
    angular_distance_to_z = np.abs(normals[:, 2])
    
    # angles range along the vertical axis (0, pi/6) normal direction constraint
    
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/5) 
    
    idx_points = points[:, 2] < -1.4
    
    idx_2 = np.logical_and(idx_downsampled, idx_points)
    
    idx_2_not = np.logical_not(idx_2)
    
    ground_points = points[idx_2]
    
    up_points = points[idx_2_not] 
    
    
    o3d_ground_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
    
    o3d_up_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(up_points))
    
    plane_model, inliers = o3d_ground_cloud.segment_plane(distance_threshold=0.2,
                                      ransac_n=3,
                                      num_iterations=30)
    [a, b, c, d] = plane_model
    print(f" Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    ground_cloud = o3d_ground_cloud.select_by_index(inliers)
    ground_cloud.paint_uniform_color([1.0, 0, 0])
    
    ground_outlier_cloud =  o3d_ground_cloud.select_by_index(inliers, True)
    
    # ground_outlier.paint_uniform_color([0,0,1])
    
    # o3d_up_points.paint_uniform_color([0,0,1])
    
    ground_outlier_points = np.asarray(ground_outlier_cloud.points)
    
    ground_free_points = np.concatenate((ground_outlier_points,up_points), axis=0)
    
    ground_free_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_free_points))
    
    ground_free_cloud.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([ground_free_cloud,ground_cloud],
                              zoom=0.14000000000000001,
                              front=[0.81700052155396041, -0.4162185016057996, 0.39908934676533259],
                              lookat=[1.5905667559463457, 0.96401393152437076, -0.10698918608948139],
                              up=[-0.32138310866945108, 0.24596667963342175, 0.91444698587292061])
    
