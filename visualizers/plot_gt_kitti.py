import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pytorch3d.io import load_obj
import numpy as np
import torch
import argparse
import pdb
import pickle as pkl
from wandb_utils.wandb import WandB
from datasets.kitti.kitti_util import Calibration
from transforms import KittiPoints2Wandb, KittiLabel2Wandb, KittiLabel2KittiPoints
# from features.points_to_beams import BeamFeatures
from os import path as osp
import copy
from tqdm import tqdm




def main(args):

    name = 'Kitti-' + str(int(args.scene))
    run = WandB(
        project='Thesis',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="Viz"
        )

    point_cloud = '/Users/stephennelson/Projects/Data/Kitti/training/velodyne/' + args.scene + ".bin"
    image = '/Users/stephennelson/Projects/Data/Kitti/training/image_2/' + args.scene + '.png'
    kitti_label = '/Users/stephennelson/Projects/Data/Kitti/training/label_2/' + args.scene + ".txt"
    calib_file = '/Users/stephennelson/Projects/Data/Kitti/training/calib/' + args.scene + ".txt"
    calib = Calibration(calib_filepath=calib_file)
    

    # read in point cloud file
    with open(point_cloud, "rb") as f:
        pc = np.fromfile(f, dtype=np.float32).reshape((-1,4))
    with open(kitti_label) as label_file:
        label = label_file.readlines()
    gt_centroids = []
    gt_dims = []
    gt_labels = []
    for object in label:
        details = object[0:-1].split(" ")
        gt_dims.append([float(details[8]),float(details[9]),float(details[10])])
        gt_centroids.append([float(details[11]),float(details[12]),float(details[13])])
        gt_labels.append(details[0])
    gt_centroids = np.array(gt_centroids)
    gt_dims = np.array(gt_dims)
    gt_colors = [[0, 255, 255]]*len(gt_centroids)

    pc = torch.tensor(pc[:,0:3])
    gt_centroids = torch.tensor(calib.project_rect_to_velo(gt_centroids))
    h, w, l = gt_dims[:, 0:1], gt_dims[:, 1:2], gt_dims[:, 2:3]
    gt_dims = np.concatenate([l,w,h],axis=1)
    gt_centroids[:, 2] += h[:, 0] / 2
    gt_dims = torch.tensor(gt_dims)

    gt_boxes = run.get_bb_dict(gt_centroids, gt_dims, labels=gt_labels, colors=gt_colors)
    pc_dict = run.get_point_cloud_log(pc, boxes=gt_boxes)
    img_dict = run.get_img_log(image)
    dict_list = [pc_dict, img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--scene', default='000142', type=str, help='path to scene')
    args = parser.parse_args()
    main(args)