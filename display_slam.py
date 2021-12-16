#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:41:41 2021

@author: songming
"""

import open3d as o3d
import numpy as np
import copy
import time

def open3d_show():
    
    point_cloud = o3d.geometry.PointCloud()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Kitti', width=1853, height=1052)
    vis.add_geometry(point_cloud)

    render_option = vis.get_render_option()
    render_option.point_size = 1
    render_option.background_color = np.asarray([0, 0, 0])  
    render_option.line_width = 2
  
    # render_option.save_to_json("/home/songming/a.json")
    # to_reset_view_point = True
    
    for i in range(500):
        
        print("/home/songming/velodyne_points/data/%010d.bin"%i)
        bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
        # bin_pcd_f = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%(i+1), dtype=np.float32)
        
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        # points_f = bin_pcd_f.reshape((-1, 4))[:, 0:3]
        
        
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # o3d_pcd_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_f))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3, origin=[0, 0, 0])
        vis.clear_geometries()
        
        ctr = vis.get_view_control()
        
        param = o3d.io.read_pinhole_camera_parameters('/home/songming/full.json')
        
        vis.add_geometry(point_cloud)
        vis.add_geometry(mesh_frame)
        ctr.convert_from_pinhole_camera_parameters(param)
        
        # if to_reset_view_point:
        #     vis.reset_view_point(True)
        #     to_reset_view_point = False
            
        vis.poll_events()
        vis.update_renderer()
        
        
        
def open3d_save():        
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1853, height=1052)
    
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%1, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    vis.add_geometry(pcd)
    
    vis.run() # user changes the view and press "q" to terminate
    
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    # print(param)
    # print(param.intrinsic.intrinsic_matrix)
    # print(param.extrinsic)
    
    o3d.io.write_pinhole_camera_parameters('/home/songming/full.json', param)
    
    vis.destroy_window()
        

def open3d_view(): 
    
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%1, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1853, height=1052)

    
    ctr = vis.get_view_control()
    
    param = o3d.io.read_pinhole_camera_parameters('/home/songming/full.json')
    
    vis.add_geometry(pcd)
   
    
    ctr.convert_from_pinhole_camera_parameters(param)
    
    vis.run()
    vis.destroy_window()
    
def show_frame():
    
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1853, height=1052)
    vis.add_geometry(mesh_frame)
    vis.run()
    vis.destroy_window()

def load_point_clouds(voxel_size):
    pcds = []
    for i in range(4,20):
        
        bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)     
        pcds.append(pcd_down)
        
    return pcds



def load_point_clouds_raw():
    pcds = []
    for i in range(4,20):
        
        bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
           
        pcds.append(pcd)
        
    return pcds

voxel_size = 0.1
mesh_list = []

pcds_down = load_point_clouds(voxel_size) #return the list containing individual pcds

pose_graph = o3d.io.read_pose_graph('/home/songming/kitti_pose.json')
pcds_raw = load_point_clouds_raw()

for point_id in range(len(pcds_raw)):
  pcds_raw[point_id].transform(pose_graph.nodes[point_id].pose)
  mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
  mesh_t = copy.deepcopy(mesh).transform(pose_graph.nodes[point_id].pose)
  mesh_list.append(mesh_t)
  
for point_id in range(len(pcds_down)):
  pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

vis = o3d.visualization.Visualizer()


vis.create_window(width=1853, height=1052)

ctr = vis.get_view_control()

param = o3d.io.read_pinhole_camera_parameters('/home/songming/full.json')
 

vis.create_window()

render_option = vis.get_render_option()
render_option.point_size = 1
render_option.background_color = np.asarray([0, 0, 0])  
render_option.line_width = 2



for point_id in range(len(pcds_down)):
    
    tic = time.time()
    vis.add_geometry(pcds_down[point_id])
    vis.add_geometry(mesh_list[point_id]) 
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    
    vis.capture_screen_image("/home/songming/pics/temp_%04d.jpg" % point_id)
    time.sleep(0.5+ tic - time.time())
    
vis.destroy_window()
    
# for point_id in range(len(pcds_down)):
#     o3d.visualization.draw_geometries(pcds_down[0:point_id],
#                                       zoom=0.561,
#                                       front=[0.19391105022671565, -0.51997936996456873, 0.83187737041659127],
#                                       lookat=[13.216348188845819, 9.0655978667509594, -11.312674020904495],
#                                       up=[0.01083534617986362, 0.84905612773617478, 0.52819152513725087])
    
    

