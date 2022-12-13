import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from pytorch3d.io import load_obj
import numpy as np
import torch
import argparse
import json
import pdb
import pickle as pkl
from wandb_utils.wandb import WandB
from transforms import NormalizeShift, NegZForward2YForward, MirrorX, ShufflePoints, KittiPoints2Wandb
from features.points_to_beams import BeamFeatures
from os import path as osp
import mmcv
from mmcv.image import tensor2imgs
import mmdet3d
import copy
from tqdm import tqdm




def main(args):

    # setting confidence threshold for prediction scores
    threshold = 0.5

    # kitti classes model has been trained on
    classes = ['Car', 'Cyclist', 'Pedestrian']
    



    with open('/multiview/3d-count/Kitti/ImageSets/val.txt') as val_list:
        val_ids = val_list.readlines()
    val_id_list = []
    for id in val_ids:
        id = id[0:-1]
        val_id_list.append(id)
    if str(args.scene) in val_id_list:
        val = int(val_id_list.index(args.scene))
    else:
        print("Not in Val Set")
        return



    name = 'Kitti-' + str(int(args.scene))
    run = WandB(
        project='3DViz',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="Vizualize"
        )

    point_cloud = '/multiview/3d-count/Kitti/training/velodyne/' + args.scene + ".bin"
    image = '/multiview/3d-count/3d-dataset-exploration/visualizations/classes/kitti/' + args.scene + '.png'
    kitti_info = '/multiview/3d-count/Kitti/kitti_infos_val.pkl'

    # reading in prediction file
    with open("/multiview/3d-count/Kitti/pp_result.pkl",'rb') as result:
        pred_data = pkl.load(result)

    # read in point cloud file
    with open(point_cloud, "rb") as f:
        pc = np.fromfile(f, dtype=np.float32).reshape((-1,4))
    # pdb.set_trace()
    # reading in ground truth files
    with open(kitti_info,'rb') as info:
        gt_data = pkl.load(info)

    pred_labels = []
    gt_labels = []

    # create masks for filtering out classes not trained
    temp_mask = [True if gt_data[val]['annos']['name'][i] != 'DontCare' else False for i in range(len(gt_data[val]['annos']['name']))]
    gt_mask = [True if gt_data[val]['annos']['name'][temp_mask][i] in classes else False for i in range(len(gt_data[val]['annos']['name'][temp_mask]))]

    gt_centroids = torch.tensor(gt_data[val]['annos']['gt_boxes_lidar'][:,0:3][gt_mask])
    gt_dims = torch.tensor(gt_data[val]['annos']['gt_boxes_lidar'][:,3:6][gt_mask])

    for k in range(len(gt_centroids)):
            gt_labels.append("           GT")

    pred_centroids = torch.tensor(pred_data[val]['boxes_lidar'][:,0:3][pred_data[val]['score'] >= threshold])
    pred_dims = torch.tensor(pred_data[val]['boxes_lidar'][:,3:6][pred_data[val]['score'] >= threshold])
    for i in range(len(pred_centroids)):
        pred_labels.append("PRED             ")

    pred_colors = [[255, 255, 0]]*len(pred_centroids)
    gt_colors = [[0, 255, 255]]*len(gt_centroids)


    swap = KittiPoints2Wandb()
    pc = swap(torch.tensor(pc[:,0:3]))
    # pdb.set_trace()
    gt_centroids = swap(gt_centroids)
    gt_dims = swap(gt_dims)
    pred_centroids = swap(pred_centroids)
    pred_dims = swap(pred_dims)

    gt_boxes = run.get_bb_dict(gt_centroids, gt_dims, labels=gt_labels, colors=gt_colors)
    pred_boxes = run.get_bb_dict(pred_centroids, pred_dims, labels=pred_labels, colors=pred_colors)
    final_boxes = np.concatenate([gt_boxes,pred_boxes])
    pc_dict = run.get_point_cloud_log(pc, boxes=final_boxes)
    img_dict = run.get_img_log(image)
    dict_list = [pc_dict, img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    # pdb.set_trace()
    scene_error = abs(len(gt_centroids) - len(pred_centroids))
    log_dict.update({'scene_erorr':scene_error})
    run.log(log_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--scene', type=str, help='path to scene')
    args = parser.parse_args()
    main(args)