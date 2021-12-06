
## Step 1. Make fragments: build local geometric surfaces (referred to as fragments)  
## from short subsequences of the input RGBD sequence. 
## This part uses RGBD Odometry, Multiway registration, and RGBD integration.

## how to create fragments (local geometric surfaces, Surface Reconstruction with Alpha shapes) from the LiDAR scan??

# Step 2. Register fragments: the fragments are aligned in a global space to detect loop closure. 
# This part uses Global registration, ICP registration, and Multiway registration.

# Step 3. Refine registration: the rough alignments are aligned more tightly. 
# This part uses ICP registration, and Multiway registration.

# Step 4. Integrate scene: integrate RGB-D images to generate a mesh model for the scene. 
# This part uses RGBD integration.

import open3d as o3d
import numpy as np
import copy

voxel_size = 0.1

def load_point_clouds(voxel_size):
    pcds = []
    for i in range(4,10):
        
        bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)     
        pcds.append(pcd_down)
        
    return pcds

def pairwise_registration(source, target, voxel_size):
    
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= voxel_size*2, max_nn=30))
    
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    transformation_icp = icp_fine.transformation
    
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    return transformation_icp, information_icp

def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id],voxel_size)
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph




pcds_down = load_point_clouds(voxel_size)
print("Point clouds visualization before the pose graph...")
o3d.visualization.draw_geometries(pcds_down,
                                  zoom=0.3612,
                                  front=[0.0274, -0.4184, 0.9078],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0897, 0.9034, 0.4192])
print("Full registration ...")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down)
    
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=voxel_size*1.5,
    edge_prune_threshold=0.25,
    reference_node=0)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
print("Transform points and display")

pcds_down_temp = copy.deepcopy(pcds_down)
pcd_combined = o3d.geometry.PointCloud()


for point_id in range(len(pcds_down)):
    print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds_down_temp[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(pcds_down,
                                  zoom=0.3612,
                                  front=[0.0274, -0.4184, 0.9078],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0897, 0.9034, 0.4192])
alpha = 0.1
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcd_combined, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


####################test part################################################
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])
    
    
# i =3
# bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
# bin_pcd_f = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%(i+1), dtype=np.float32)

# points = bin_pcd.reshape((-1, 4))[:, 0:3]
# points_f = bin_pcd_f.reshape((-1, 4))[:, 0:3]


# o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

# pcd_down = o3d_pcd.voxel_down_sample(0.1)
# pcd_down.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= 0.1*2, max_nn=30))


# o3d_pcd_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_f))
    

# o3d.visualization.draw_geometries([o3d_pcd])
##############################Alpha shapes###############################
# alpha = 0.1
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#     o3d_pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

###############################Ball pivoting#######################


# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#                o3d_pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([o3d_pcd, rec_mesh])

#################Poisson surface reconstruction#########################

# print(o3d_pcd)
# o3d.visualization.draw_geometries([o3d_pcd], zoom=0.664,
#                                   front=[-0.4761, -0.4698, -0.7434],
#                                   lookat=[1.8900, 3.2596, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])

# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=20)
# print(mesh)
# o3d.visualization.draw_geometries([mesh], zoom=0.664,
#                                   front=[-0.4761, -0.4698, -0.7434],
#                                   lookat=[1.8900, 3.2596, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])


#####################################################################################################################


