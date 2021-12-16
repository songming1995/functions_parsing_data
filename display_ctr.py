#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:37:33 2021

@author: songming
"""

import open3d as o3d
import numpy as np

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
    
    
        

if __name__ == "__main__":
    # open3d_save()
    # open3d_view()
    open3d_show()
    # show_frame()
