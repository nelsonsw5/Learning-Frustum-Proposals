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
from transforms import NormalizeShift, NegZForward2YForward, MirrorX, ShufflePoints
from features.points_to_beams import BeamFeatures
from os import path as osp
import mmcv
from mmcv.image import tensor2imgs
import mmdet3d
from bound_box import ThreeDimBoundBox

def main(args):
    threshold = 0.5
    classes = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub']
    name = 'SunRGBD-' + str(int(args.scene))
    run = WandB(
        project='3DViz',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="Vizualize"
        )

    image_path = '/multiview/3d-count/3d-dataset-exploration/visualizations/sunrgbd/'



    points = '/multiview/3d-count/OFFICIAL_SUNRGBD/sunrgbd/results/' + args.scene + "/" + args.scene + "_points.obj"
    

    pc, _, _ = load_obj(points)
    pc[:,0] = pc[:,0] * -1

    ### Ground Truth
    gt_labels = []
    gt_centroids = []
    gt_dims = []
    gts = '/multiview/3d-count/OFFICIAL_SUNRGBD/sunrgbd/results/' + args.scene + '/' + str(args.scene) + '_gt.obj'
    gts, _, _ = load_obj(gts)
    gts[:,0] = gts[:,0] * -1
    gt_boxes = []
    for box in np.split(gts,len(gts)/8):
        bb = ThreeDimBoundBox(corners=np.array(box))
        box_arr = bb.get_corners().transpose()
        box_dict = {
                    "corners": list(zip(*box_arr.tolist())),
                    "color": [0, 255, 0],  
                    "label": "GT"
                }
        gt_boxes.append(box_dict)


    ### Prediction
    pred_labels = []
    pred_centroids = []
    pred_dims = []
    pred_points = '/multiview/3d-count/OFFICIAL_SUNRGBD/sunrgbd/results/' + args.scene + '/' + str(args.scene) + '_pred.obj'
    preds, _, _ = load_obj(pred_points)
    preds[:,0] = preds[:,0] * -1
    pred_boxes = []
    for box in np.split(preds,len(preds)/8):       
        bb = ThreeDimBoundBox(corners=np.array(box))
        box_arr = bb.get_corners().transpose()
        box_dict = {
                    "corners": list(zip(*box_arr.tolist())),
                    "color": [0, 255, 255],  # green boxes if none given
                    "label": "PRED"
                }
        pred_boxes.append(box_dict)
        

    gt_num = len(gt_boxes)
    pred_num = len(pred_boxes)
    gt_boxes = run.get_bb_dict(boxes=gt_boxes)
    pred_boxes = run.get_bb_dict(boxes=pred_boxes)
    final_boxes = np.concatenate([gt_boxes,pred_boxes])
    pc_dict = run.get_point_cloud_log(pc, boxes=final_boxes)
    image = image_path + args.scene + '.jpg'
    img_dict = run.get_img_log(image)
    dict_list = [pc_dict, img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    scene_error = abs(gt_num - pred_num)
    log_dict.update({'scene_erorr':scene_error})
    run.log(log_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--scene', type=str, help='path to scene')
    args = parser.parse_args()
    main(args)