#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 07:59:01 2021

@author: songming
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:00:38 2021

@author: songming
"""
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
## http://localhost:8888/notebooks/Open3D/examples/python/pipelines/global_registration.ipynb
## registration_ransac_based_on_feature_matching
## registration_fast_based_on_feature_matching
import os, os.path

## detect the ground points and visualize, remove them 

## visualize the bounding box, specify the rendering color

## visualize the ISS keypoints with KITTI, CTR + C -> CTR + V, json file

## specify different view perspectives (visualization), specify the point size -> done

## bbox range -> point index-> uniform paint dynamic points in red others in blue

## filter out scattered outliers -> statistical outlier -> done

## benchmark semantic kitti -> to do field of view  

## full screen width and height, kitti intrinsic and extrinsic, visual validation
## visualize the original frame ->  done

## keyboard waitkey for visulization

# class of PinholeCameraParameters
# class of ViewTrajectory


# Load data
# pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
# vol = o3d.visualization.read_selection_polygon_volume(
#     "../../test_data/Crop/cropped.json")
# chair = vol.crop_point_cloud(pcd)

# dists = pcd.compute_point_cloud_distance(chair)
# dists = np.asarray(dists)
# ind = np.where(dists > 0.01)[0]
# pcd_without_chair = pcd.select_by_index(ind)
# o3d.visualization.draw_geometries([pcd_without_chair],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])


######  multiway_registration  ########
### TransformationEstimationPointToPlane and TransformationEstimationColoredICP 
### require pre-computed normal vectors for target PointCloud.
def pairwise_registration(source, target):
    # print("Apply point-to-plane ICP")
    voxel_size = 0.02
    max_correspondence_distance_coarse =  voxel_size * 15
    max_correspondence_distance_fine =  voxel_size * 1.5
    
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(icp_fine)
    
    transformation_icp = icp_fine.transformation
    
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    return transformation_icp, information_icp

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    
    # estimate a set of FPFH features for all the downsampled points in the input dataset.
    # persistent FPFH points
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("Since the downsampling voxel size is %.3f," % voxel_size)
    print("we use a liberal distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# voxel_size = 0.05  # means 5cm for this dataset
# source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
#     voxel_size)

DIR = '/home/songming/velodyne_points/data'
num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print (num_iter)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
num_iter = 1
for i in range(num_iter):
    print("/home/songming/velodyne_points/data/%010d.bin"%i)
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    bin_pcd_f = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%(i+1), dtype=np.float32)
    
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    points_f = bin_pcd_f.reshape((-1, 4))[:, 0:3]
    
    
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_f))
    
    source = o3d_pcd.voxel_down_sample(voxel_size=0.04)
    target = o3d_pcd_f.voxel_down_sample(voxel_size=0.04)
    
    cl, ind = source.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2)
    print(ind)
    ## ISS keypoint detection 
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(cl)
    cl.paint_uniform_color([1.0, 0.0, 0.0])
    keypoints.paint_uniform_color([0.0, 0.0, 1.0])
    
    # vis = o3d.visualization.Visualizer()
    # render_option = vis.get_render_option()
   
    o3d.visualization.draw_geometries([cl, keypoints], zoom=0.11999999999999995,
                                  front=[ -0.80266402037269768, -0.34524298906839201, 0.4863514664296385  ],
                                  lookat=[ -1.5211321862471525, 1.7305793834663294, -4.6735039146346899 ],
                                  up=[ 0.38317142696032985, 0.32640703986290465, 0.86408223097638159 ])
    

    
    target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    
    transformation_icp, information_icp = pairwise_registration(source, target)

#point to point ICP    
    # threshold = 0.02 #or 0.2
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                       [-0.139, 0.967, -0.215, 0.7],
    #                       [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    # source, target, threshold, trans_init,    
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # ### fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
    # ### inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    
    
#point to plane ICP    specify the convergence criterti
    # threshold = 0.2
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                       [-0.139, 0.967, -0.215, 0.7],
    #                       [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    # source, target, threshold, trans_init,    
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # ### fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
    # ### inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    
    
###################################
#     if i == 0:
#         vis.add_geometry(source)
#         # print("initialization finished")
#     else:
#         vis.clear_geometries()
#         vis.add_geometry(source)
        
#         # print("update finished")
        
#     vis.poll_events()
#     vis.update_renderer()
    
# vis.destroy_window()


# A = o3d.geometry.AxisAlignedBoundingBox(np.array([1,2,3]), np.array([10,11,13]))
# o3d.visualization.draw_geometries([A],zoom=5)

######################################################### ground points segmentation

# import os
# os.environ['NUMEXPR_MAX_THREADS'] = '16'

# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# import copy
# import sys
# import os, os.path

## http://localhost:8888/notebooks/Open3D/examples/python/pipelines/global_registration.ipynb
## registration_ransac_based_on_feature_matching
## registration_fast_based_on_feature_matching

## detect the ground points and visualize, remove them 

## visualize the bounding box, specify the rendering color

## visualize the ISS keypoints with KITTI, CTR + C -> CTR + V, json file

## specify different view perspectives (visualization), specify the point size -> done

## bbox range -> point index-> uniform paint dynamic points in red others in blue

## filter out scattered outliers -> statistical outlier -> done

## benchmark semantic kitti -> to do -> check field of view (overlapped with camera view??)  

## full screen width and height, kitti intrinsic and extrinsic, visual validation
## visualize the original frame ->  done

## keyboard waitkey for visulization

# class of PinholeCameraParameters
# class of ViewTrajectory


# Load data
# pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
# vol = o3d.visualization.read_selection_polygon_volume(
#     "../../test_data/Crop/cropped.json")
# chair = vol.crop_point_cloud(pcd)

# dists = pcd.compute_point_cloud_distance(chair)
# dists = np.asarray(dists)
# ind = np.where(dists > 0.01)[0]
# pcd_without_chair = pcd.select_by_index(ind)
# o3d.visualization.draw_geometries([pcd_without_chair],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])


######  multiway_registration  ########
### TransformationEstimationPointToPlane and TransformationEstimationColoredICP 
### require pre-computed normal vectors for target PointCloud.






# def pairwise_registration(source, target):
#     # print("Apply point-to-plane ICP")
#     voxel_size = 0.02
#     max_correspondence_distance_coarse =  voxel_size * 15
#     max_correspondence_distance_fine =  voxel_size * 1.5
    
#     icp_coarse = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_coarse, np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
#     icp_fine = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_fine,
#         icp_coarse.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     print(icp_fine)
    
#     transformation_icp = icp_fine.transformation
    
#     information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
#         source, target, max_correspondence_distance_fine,
#         icp_fine.transformation)
    
#     return transformation_icp, information_icp

# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = pcd.voxel_down_sample(voxel_size)

#     radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#     radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    
#     # estimate a set of FPFH features for all the downsampled points in the input dataset.
#     # persistent FPFH points
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

# def prepare_dataset(voxel_size):
#     print(":: Load two point clouds and disturb initial pose.")
#     source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
#     target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)
#     # draw_registration_result(source, target, np.identity(4))

#     source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#     return source, target, source_down, target_down, source_fpfh, target_fpfh


# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("Since the downsampling voxel size is %.3f," % voxel_size)
#     print("we use a liberal distance threshold %.3f." % distance_threshold)
    
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
#     return result

# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
#     distance_threshold = voxel_size * 0.4
#     print(":: Point-to-plane ICP registration is applied on original point")
#     print("   clouds to refine the alignment. This time we use a strict")
#     print("   distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, distance_threshold, result_ransac.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     return result

# # voxel_size = 0.05  # means 5cm for this dataset
# # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
# #     voxel_size)

# DIR = '/home/songming/velodyne_points/data'
# num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# print (num_iter)

# # vis = o3d.visualization.Visualizer()
# # vis.create_window()
# num_iter = 10

# for i in range(num_iter):
    
#     print("/home/songming/velodyne_points/data/%010d.bin"%i)
    
#     bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    
#     points = bin_pcd.reshape((-1, 4))[:, 0:3] 
    
#     o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
#     ##The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. 
#     ##It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
    
#     o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=10))
    
#     normals = np.asarray(o3d_pcd.normals)
   
#     angular_distance_to_z = np.abs(normals[:, 2])
    
#     # angles range along the vertical axis (0, pi/6) normal direction constraint
    
#     idx_downsampled = angular_distance_to_z > np.cos(np.pi/5) 
    
#     idx_points = points[:, 2] < -1.4
    
#     idx_2 = np.logical_and(idx_downsampled, idx_points)
    
#     idx_2_not = np.logical_not(idx_2)
    
#     ground_points = points[idx_2]
    
#     up_points = points[idx_2_not] 
    
    
#     o3d_ground_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_points))
    
#     o3d_up_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(up_points))
    
#     plane_model, inliers = o3d_ground_cloud.segment_plane(distance_threshold=0.2,
#                                       ransac_n=3,
#                                       num_iterations=30)
#     [a, b, c, d] = plane_model
#     print(f" Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
#     ground_cloud = o3d_ground_cloud.select_by_index(inliers)
#     ground_cloud.paint_uniform_color([1.0, 0, 0])
    
#     ground_outlier_cloud =  o3d_ground_cloud.select_by_index(inliers, True)
    
#     # ground_outlier.paint_uniform_color([0,0,1])
    
#     # o3d_up_points.paint_uniform_color([0,0,1])
    
#     ground_outlier_points = np.asarray(ground_outlier_cloud.points)
    
#     ground_free_points = np.concatenate((ground_outlier_points,up_points), axis=0)
    
#     ground_free_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_free_points))
    
#     ground_free_cloud.paint_uniform_color([0,0,1])
    
#     o3d.visualization.draw_geometries([ground_free_cloud,ground_cloud],
#                               zoom=0.14000000000000001,
#                               front=[0.81700052155396041, -0.4162185016057996, 0.39908934676533259],
#                               lookat=[1.5905667559463457, 0.96401393152437076, -0.10698918608948139],
#                               up=[-0.32138310866945108, 0.24596667963342175, 0.91444698587292061])
    
    

######################################################################################################
# points = np.asarray(pcd.points)
# mask = points[:,1] > y_threshold
# pcd.points = o3d.utility.Vector3dVector(points[mask]) # normals and colors are unchanged

# # alternative
# pcd = pcd.select_by_index(np.where(points[:,1] > y_theshold)[0])

#####################################################################################################

# pcd = o3d.io.read_point_cloud("pointcloud.pcd")
# z_threshold =4
# points = np.asarray(pcd.points)
# mask = points[:,2] > z_threshold
# pcd.points = o3d.utility.Vector3dVector(points[mask]) # normals and colors are unchanged

# # alternative
# pcd = pcd.select_by_index(np.where(points[:,2] > z_threshold)[0])


# o3d.visualization.draw_geometries([pcd])

#######################################################################################################
# import numpy as np
# import open3d as o3d

# # create test data
# points = np.random.uniform(-1, 1, size=(100, 3))
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # threshold data
# points = np.asarray(pcd.points)
# pcd_sel = pcd.select_by_index(np.where(points[:, 2] > 0)[0])

# # visualize different point clouds
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd_sel])

#######################################################################################################
# import open3d as o3d
# import numpy as np

# #Create two random points
# randomPoints = np.random.rand(2, 3)

# pointSet = o3d.geometry.PointCloud()

# pointSet.points = o3d.utility.Vector3dVector(randomPoints)

# #Visualize the two random points
# o3d.visualization.draw_geometries([pointSet])

# #Here I want to add more points to the pointSet
# #This solution does not work effective

# #Create another random set
# p1 = np.random.rand(3, 3)

# p2 = np.concatenate((pointSet.points, p1), axis=0)

# pointSet2 = o3d.geometry.PointCloud()

# pointSet2.points = o3d.utility.Vector3dVector(p2)

# o3d.visualization.draw_geometries([pointSet2])

##########################################################################################



    # downsampled = o3d_pcd.select_by_index([0 ,1, 2, 3, 5, 20 ,10])
        
# pcd = o3d.io.read_point_cloud("../../test_data/fragment.pcd")
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                  ransac_n=3,
#                                  num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                           zoom=0.8,
#                           front=[-0.4999, -0.1659, -0.8499],
#                           lookat=[2.1813, 2.0619, 2.0999],
#                           up=[0.1204, -0.9852, 0.1215])


    # source = o3d_pcd.voxel_down_sample(voxel_size=0.04)

    
    # cl, ind = source.remove_statistical_outlier(nb_neighbors=20,
    #                                                 std_ratio=2)
    # ## ISS keypoint detection 
    # keypoints = o3d.geometry.keypoint.compute_iss_keypoints(cl)
    # cl.paint_uniform_color([1.0, 0.0, 0.0])
    # keypoints.paint_uniform_color([0.0, 0.0, 1.0])
    
    # # vis = o3d.visualization.Visualizer()
    # # render_option = vis.get_render_option()
   
    # o3d.visualization.draw_geometries([cl, keypoints], zoom=0.11999999999999995,
    #                               front=[ -0.80266402037269768, -0.34524298906839201, 0.4863514664296385  ],
    #                               lookat=[ -1.5211321862471525, 1.7305793834663294, -4.6735039146346899 ],
    #                               up=[ 0.38317142696032985, 0.32640703986290465, 0.86408223097638159 ])
    

    
    # target.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    
    # transformation_icp, information_icp = pairwise_registration(source, target)

#point to point ICP    
    # threshold = 0.02 #or 0.2
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                       [-0.139, 0.967, -0.215, 0.7],
    #                       [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    # source, target, threshold, trans_init,    
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # ### fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
    # ### inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    
    
#point to plane ICP    specify the convergence criterti
    # threshold = 0.2
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                       [-0.139, 0.967, -0.215, 0.7],
    #                       [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    # source, target, threshold, trans_init,    
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # ### fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
    # ### inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    
    
###################################
#     if i == 0:
#         vis.add_geometry(source)
#         # print("initialization finished")
#     else:
#         vis.clear_geometries()
#         vis.add_geometry(source)
        
#         # print("update finished")
        
#     vis.poll_events()
#     vis.update_renderer()
    
# vis.destroy_window()


# A = o3d.geometry.AxisAlignedBoundingBox(np.array([1,2,3]), np.array([10,11,13]))
# o3d.visualization.draw_geometries([A],zoom=5)
