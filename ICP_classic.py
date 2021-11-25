import os
import os.path
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
## http://localhost:8888/notebooks/Open3D/examples/python/pipelines/global_registration.ipynb
## registration_ransac_based_on_feature_matching
## registration_fast_based_on_feature_matching



## detect the ground points and visualize, remove them 

## visualize the bounding box, specify the rendering color

## visualize the ISS keypoints with KITTI, CTR + C -> CTR + V, json file

## specify different view perspectives (visualization), specify the point size -> done

## bbox range -> point index-> uniform paint dynamic points in red others in blue

## filter out scattered outliers -> statistical outlier -> done

## benchmark semantic kitti -> to do -> field of view  
## specify the point cloud range for ICP, improve the efficiency
## remove the ground points beforehand

## full screen width and height, kitti intrinsic and extrinsic, visual validation
## visualize the original frame ->  done

## keyboard waitkey for visulization -> to be done

# class of PinholeCameraParameters
# class of ViewTrajectory


### TransformationEstimationPointToPlane and TransformationEstimationColoredICP 
### require pre-computed normal vectors for target PointCloud.

def pairwise_registration(source, target, voxel_size, last_itertion_pose):
    ## robust kernel
    # loss = o3d.pipelines.registration.HuberLoss(k=0.1)
    # print("Using robust loss:", loss)
    # p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)


    max_correspondence_distance_coarse =  voxel_size * 15
    max_correspondence_distance_fine =  voxel_size * 2
    
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, last_itertion_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    
    transformation_icp = icp_fine.transformation
    

    correspondence_set = np.asarray(icp_fine.correspondence_set)
    
    inliers_source = correspondence_set[:,0]
    outlier_source =  source.select_by_index(inliers_source, True)
    
    inliers_target = correspondence_set[:,1]
    outlier_target =  target.select_by_index(inliers_target, True)

    
    # information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    #     source, target, max_correspondence_distance_fine,
    #     icp_fine.transformation)
    
    return transformation_icp, correspondence_set, outlier_source, outlier_target

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
    
    ### he most important hyperparameter of this function is RANSACConvergenceCriteria. 
    ### It defines the maximum number of RANSAC iterations and the confidence probability. 
    ### The larger these two numbers are, the more accurate the result is, but also the more time the algorithm takes.
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

########################################################################
def draw_registration_result(source, target, transformation, outliers_target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    outliers_target.paint_uniform_color([1, 0, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, outliers_target],
                                      zoom=0.175,
                                      front=[ -0.64264410656652293, -0.65200905363026951, 0.40235897688441846 ],
                                      lookat=[ 1.9892000000000001, 2.0207999999999999, 1.8945000000000001 ],
                                      up=[ 0.34201846275438874, 0.22579518642047891, 0.91216221415078702 ])


def draw_registration_outlier( target, outliers_target):

    target_temp = copy.deepcopy(target)

    target_temp.paint_uniform_color([0, 0.651, 0.929])
    outliers_target.paint_uniform_color([1, 0, 0])
 
    o3d.visualization.draw_geometries([ target_temp, outliers_target],
                                      zoom=0.175,
                                      front=[ -0.64264410656652293, -0.65200905363026951, 0.40235897688441846 ],
                                      lookat=[ 1.9892000000000001, 2.0207999999999999, 1.8945000000000001 ],
                                      up=[ 0.34201846275438874, 0.22579518642047891, 0.91216221415078702 ])
#####################################################################

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
    ## increase the distance to incorporate more ground points
    plane_model, inliers = o3d_ground_cloud.segment_plane(distance_threshold=0.3,
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
#####################################################################


num_iter = 5
last_pose = np.identity(4)
for i in range(num_iter):
    print("/home/songming/velodyne_points/data/%010d.bin"%i)
    
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    bin_pcd_f = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%(i+1), dtype=np.float32)
    
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    points_f = bin_pcd_f.reshape((-1, 4))[:, 0:3]
    
    
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d_pcd_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_f))
    
    o3d_pcd_gf = remove_ground(o3d_pcd)
    o3d_pcd_f_gf = remove_ground(o3d_pcd_f)
    
    voxel = 0.2
    
    source = o3d_pcd_gf.voxel_down_sample(voxel_size=voxel)
    target = o3d_pcd_f_gf.voxel_down_sample(voxel_size=voxel)
    
    # std_ratio, which allows setting the threshold level based on the standard deviation of the 
    # average distances across the point cloud. 
    # The lower this number the more aggressive the filter will be.
    cl_source, ind_source = source.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2)

    cl_target, ind_target = target.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2)
    
    # cl_source.estimate_normals(
    #  search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel, max_nn=30))
    # cl_target.estimate_normals(
    #  search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel, max_nn=30))
    
    # ## ISS keypoint detection, using defaut values
    # keypoints_source = o3d.geometry.keypoint.compute_iss_keypoints(cl_source)
    # keypoints_target = o3d.geometry.keypoint.compute_iss_keypoints(cl_target)
    
    # ## FPFH descriptor calculation
    # radius_feature = voxel * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    # source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(keypoints_source,
    # o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(keypoints_source,
    # o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # ## ransac ICP with FPFH descriptor
    # result_ransac = execute_global_registration(keypoints_source, keypoints_target,
    #                                         source_fpfh, target_fpfh,
    #                                         voxel)
    # draw_registration_result(keypoints_source, keypoints_target, result_ransac.transformation)
    
    ###################################################################################
   
    ### remove ground points, see the results-> done
    ### see the simple pairwise_registration overlapping results -> done
    ## fix the initial value problem -> done
    ## return and visualize the ICP outliers -> done
    ######################################################################
   
    cl_target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= voxel*2, max_nn=30))
    # print('last pose is')
    # print(last_pose)
    transformation_icp, set, outliers_source, outliers_target = pairwise_registration(cl_source, cl_target, voxel, last_pose)
    last_pose  = transformation_icp
    # draw_registration_result(source, target, np.identity(4))
    # draw_registration_result(cl_source, cl_target, transformation_icp, outliers_target)
    
    draw_registration_outlier(cl_target, outliers_target)
    
    # print(transformation_icp)


    # ### fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). 
    # ### fitness  The higher the better.
    
    # ### inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.

    
    



# A = o3d.geometry.AxisAlignedBoundingBox(np.array([1,2,3]), np.array([10,11,13]))
# o3d.visualization.draw_geometries([A],zoom=5)

######################################################### ground points segmentation




## benchmark semantic kitti -> to do -> check field of view (overlapped with camera view??)  

## full screen width and height, kitti intrinsic and extrinsic, visual validation
## visualize the original frame ->  done

## keyboard waitkey for visulization

# class of PinholeCameraParameters
# class of ViewTrajectory


    
    

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





    
