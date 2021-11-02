import numpy as np
import open3d.ml as _ml3d
import math

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from open3d.ml.datasets import KITTI
from open3d._ml3d.datasets.utils import BEVBox3D
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for inference of object detection')
    parser.add_argument('--framework',
                        help='deep learning framework: tf or torch',default='torch')
    parser.add_argument('--path_kitti', help='path to KITTI', default = '/home/songming/Downloads/Kitti' )
    parser.add_argument('--path_ckpt_pointpillars',
                        help='path to PointPillars checkpoint', default= '/home/songming/Open3D-ML/logs/pointpillars_kitti_202012221652utc.pth')
    parser.add_argument('--device',
                        help='device to run the pipeline',
                        default='gpu')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args



def main(args):

    framework = _ml3d.utils.convert_framework_name(args.framework)
    args.device = _ml3d.utils.convert_device_name(args.device)
    if framework == 'torch':
        from open3d.ml.torch.dataloaders import TorchDataloader as Dataloader
        
    else:
        import tensorflow as tf
        import open3d.ml.tf as ml3d

        from ml3d.tf.dataloaders import TFDataloader as Dataloader

        device = args.device
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    ObjectDetection = _ml3d.utils.get_module("pipeline", "ObjectDetection",
                                             framework)
    PointPillars = _ml3d.utils.get_module("model", "PointPillars", framework)
    cfg = _ml3d.utils.Config.load_from_file(
        "/home/songming/Open3D-ML/ml3d/configs/pointpillars_kitti.yml")

    model = PointPillars(device=args.device, **cfg.model)
    dataset = KITTI(args.path_kitti)
   
    pipeline = ObjectDetection(model, dataset, device=args.device)

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=args.path_ckpt_pointpillars)

    test_split = Dataloader(dataset=dataset.get_split('testing'),
                            preprocess=model.preprocess,
                            transform=None,
                            use_cache=False,
                            shuffle=False)
    data = test_split[5]['data'] #check !!!
    # run inference on a single example.
    result = pipeline.run_inference(data)[0]
    
    print(result)
    
    # boxes = data['bbox_objs']
    # boxes.extend(result)

    vis = Visualizer()

    lut = LabelLUT()
    for val in sorted(dataset.label_to_names.keys()):
        lut.add_label(val, val)

    vis.visualize([{
        "name": "KITTI",
        'points': data['point']
    }], 
                   lut,
                   bounding_boxes=result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
