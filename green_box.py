#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 18:31:46 2021
@author: songming
"""
# check bounding box in /home/songming/Open3D-ML/ml3d/vis/visualizer.py
# check bounding box in /home/songming/Open3D-ML/ml3d/datasets/utils/bev_box.py
# https://github.com/isl-org/Open3D-ML/blob/master/ml3d/datasets/utils/bev_box.py
#https://github.com/isl-org/Open3D-ML/blob/d58b24edd37de7889446360164cd5500e0bde060/ml3d/vis/boundingbox.py
# self.bounding_box_data.append()
import numpy as np
import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d.datasets.utils import BEVBox3D
# from open3d._ml3d.vis import BoundingBox3D

import os, os.path
DIR = '/home/songming/Music/Kitti/testing/velodyne'
num_iter = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


cfg_file = "/home/songming/Open3D-ML/ml3d/configs/pointpillars_kitti.yml" # check the cfg files
cfg = _ml3d.utils.Config.load_from_file(cfg_file)


model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/songming/Downloads/Kitti"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# # download the weights.
ckpt_folder = "/home/songming/Open3D-ML/logs/"

os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)
test_split = dataset.get_split("testing")
vis = o3d.visualization.Visualizer()


vis.create_window()

for i in range(100):
    
    data = test_split.get_data(i)
    # print('the data is like')
    points = data['full_point'][:,0:3]

    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down_points = o3d_pcd.voxel_down_sample(voxel_size=0.001)


# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
    result = pipeline.run_inference(data)

#print(result[0][0]) 
#print(result[0][1])
# print('the result is like')
# print(result)    #BEVBox3D object open3d._ml3d.datasets.utils.bev_box.BEVBox3D

    lines = BEVBox3D.create_lines(result[0])
    
    line_colors = [[0, 1, 0] for i in range(len(lines.lines))]
    
    lines.colors = o3d.utility.Vector3dVector(line_colors)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=3, origin=[0, 0, 0])
    
    vis.clear_geometries()
    
    ctr = vis.get_view_control()
    
    param = o3d.io.read_pinhole_camera_parameters('/home/songming/full.json')
    
    vis.add_geometry(lines)
    vis.add_geometry(down_points)
    vis.add_geometry(mesh_frame)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])  
    render_option.line_width = 5.0 # has no effect in the visualization window
    ctr.convert_from_pinhole_camera_parameters(param)
    
    
    vis.poll_events()
    vis.update_renderer()
# vis.run()

vis.destroy_window()

# using dataset loader's save function in whatever format 

# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
