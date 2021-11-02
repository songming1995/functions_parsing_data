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

import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d.datasets.utils import BEVBox3D
# from open3d._ml3d.vis import BoundingBox3D
cfg_file = "/home/songming/Open3D-ML/ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)


model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/home/songming/Music/Kitti"
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
test_split = dataset.get_split("test")


data = test_split.get_data(1)
print('the data is like')
points = data['full_point'][:,0:3]

o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
down_points = o3d_pcd.voxel_down_sample(voxel_size=0.2)


# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)

#print(result[0][0]) 
#print(result[0][1])
print('the result is like')
print(result)    #BEVBox3D object open3d._ml3d.datasets.utils.bev_box.BEVBox3D

lines = BEVBox3D.create_lines(result[0])
print("the process is finished")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(lines)
vis.add_geometry(down_points)
vis.run()
vis.destroy_window()

# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
