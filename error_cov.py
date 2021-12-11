#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:03:32 2021

@author: songming
"""
import open3d as o3d
import numpy as np
import time

# num_iter = 2760
num_frame = 2760
cov_xx_list = []
cov_yy_list = []
cov_zz_list = []
time_index = []
cov_xx_list_ = []
cov_yy_list_ = []
cov_zz_list_ = []
transformation_matrix = np.eye(4)
trans_z = []
trans_x = []

calib = np.loadtxt("/home/songming/Desktop/dataset_kitti/data_odometry_calib/dataset/sequences/05/calib.txt")
Tr = np.eye(4)
Tr[0:3,:] = np.reshape(calib[4],(3,4))



for i in range(num_frame):
    
    time_index.append(i)
            
    voxel_size = 0.1
    bin_pcd = np.fromfile("/home/songming/Desktop/dataset_kitti/velodyne/sequences/05/velodyne/%06d.bin"%i, dtype=np.float32)
    bin_pcd_ = np.fromfile("/home/songming/Desktop/dataset_kitti/velodyne/sequences/05/velodyne/%06d.bin"%(i+1), dtype=np.float32)
    
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    points_ = bin_pcd_.reshape((-1, 4))[:, 0:3]
    
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_))
    
    pcd_down = o3d_pcd.voxel_down_sample(0.1)
    pcd_down_ = o3d_pcd_.voxel_down_sample(0.1)
    
    pcd_down.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1*2, max_nn=30))
    pcd_down_.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1*2, max_nn=30))
    
    
    max_correspondence_distance_coarse =  voxel_size * 15
    max_correspondence_distance_fine =  voxel_size * 2
    tic = time.time()
    
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd_down_, pcd_down, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd_down_, pcd_down, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    
    
    print("the elapsed time is:" )
    print(time.time() - tic)
    # transformation_icp = icp_fine.transformation
    
    # correspondence_set = np.asarray(icp_fine.correspondence_set)
    
    # inliers_source = correspondence_set[:,0]
    # outlier_source =  source.select_by_index(inliers_source, True)
    
    # inliers_target = correspondence_set[:,1]
    # outlier_target =  target.select_by_index(inliers_target, True)
    
    transformation_matrix = np.dot(transformation_matrix, icp_fine.transformation)
    
    transformation_matrix_camera = np.dot(Tr, transformation_matrix)
    
    
    trans_x.append(transformation_matrix_camera[0][3])
    trans_z.append(transformation_matrix_camera[2][3])
    
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd_down, pcd_down_, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    cov = np.linalg.inv(information_icp)
    print('::The covariance matrix of the estimated transformation is')
    print(cov)
    
    cov_xx = np.sqrt(cov[3][3]) * 3
    cov_yy = np.sqrt(cov[4][4]) * 3
    cov_zz = np.sqrt(cov[5][5]) * 3
    
    cov_xx_list.append(cov_xx)
    cov_xx_list_.append(-cov_xx)
    
    cov_yy_list.append(cov_yy)
    cov_yy_list_.append(-cov_yy)
    
    cov_zz_list.append(cov_zz)    
    cov_zz_list_.append(-cov_zz)
    
    
    
import matplotlib.pyplot as plt    

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(time_index, cov_xx_list, label='3_sigma')  # Plot some data on the axes.
ax.plot(time_index, cov_xx_list_, label='-3_sigma')   # Plot more data on the axes...
  # ... and some more.
ax.set_xlabel('time index')  # Add an x-label to the axes.
ax.set_ylabel('uncertainty along x')  # Add a y-label to the axes.
ax.set_title("Uncertainty estimation")  # Add a title to the axes.
ax.legend()  # Add a legend. 

#####################################################################################
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(time_index, cov_yy_list, label='3_sigma')  # Plot some data on the axes.
ax.plot(time_index, cov_yy_list_, label='-3_sigma')   # Plot more data on the axes...
  # ... and some more.
ax.set_xlabel('time index')  # Add an x-label to the axes.
ax.set_ylabel('uncertainty along y')  # Add a y-label to the axes.
ax.set_title("Uncertainty estimation")  # Add a title to the axes.
ax.legend()  # Add a legend. 

####################################################################################
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(time_index, cov_zz_list, label='3_sigma')  # Plot some data on the axes.
ax.plot(time_index, cov_zz_list_, label='-3_sigma')   # Plot more data on the axes...
  # ... and some more.
ax.set_xlabel('time index')  # Add an x-label to the axes.
ax.set_ylabel('uncertainty along z')  # Add a y-label to the axes.
ax.set_title("Uncertainty estimation")  # Add a title to the axes.
ax.legend()  # Add a legend. 
    
#################################################################################
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(trans_x, trans_z, label='Camera frame pose')  # Plot some data on the axes.
# ... and some more.
ax.set_xlabel('x/m')  # Add an x-label to the axes.
ax.set_ylabel('z/m')  # Add a y-label to the axes.
ax.set_title("Odometry visualization")  # Add a title to the axes.
ax.legend()  # Add a legend. 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
