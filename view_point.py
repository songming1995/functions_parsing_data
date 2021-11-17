#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:37:33 2021

@author: songming
"""

import open3d as o3d
import numpy as np

def open3d_save():        
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=400, height=300)
    
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
    
    o3d.io.write_pinhole_camera_parameters('/home/songming/view_point.json', param)
    
    vis.destroy_window()
        

def open3d_view(): 
    
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%1, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=400, height=300)

    
    ctr = vis.get_view_control()
    
    param = o3d.io.read_pinhole_camera_parameters('/home/songming/view_point.json')
    
    vis.add_geometry(pcd)
    
    ctr.convert_from_pinhole_camera_parameters(param)
    
    vis.run()
    vis.destroy_window()
        
       

if __name__ == "__main__":
    open3d_save()
    open3d_view()
