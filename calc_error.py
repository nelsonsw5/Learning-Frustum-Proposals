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
from os import path as osp
import copy
from tqdm import tqdm

def main(args):

    


    if args.dataset == 'sunrgbd':
        classes = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub']
        with open('/multiview/3d-count/OFFICIAL_SUNRGBD/sunrgbd/results/pretrained_results.pkl',"rb") as results_file:
            predictions = pkl.load(results_file)
        with open('/multiview/3d-count/OFFICIAL_SUNRGBD/sunrgbd/sunrgbd_infos_val.pkl',"rb") as gt_file:
            gts = pkl.load(gt_file)
        thresholds = args.scores.split(",")
        # pdb.set_trace()
        for thresh in thresholds:
            thresh = float(thresh)
            prediction_count = []
            gt_count = []
            print("prediction scene count: ", len(predictions))
            print("gt scene count: ", len(gts))
            for k in tqdm(range(len(predictions))):
                gt_count.append(gts[k]['annos']['gt_num'])
                prediction_count.append(len(predictions[k]['scores_3d'][predictions[k]['scores_3d'] > float(thresh)]))
            mae = np.mean(abs(np.array(gt_count) - np.array(prediction_count)))

            mape = np.mean(np.abs((np.array(gt_count) - np.array(prediction_count)) / np.array(gt_count))) * 100
            mse = np.mean(np.square(np.array(gt_count) - np.array(prediction_count)))
            print("threshold: ", thresh)
            print("mae: ", mae)
            print("mape: ", mape)
            print("mse: ", mse)
            print()
    if args.dataset == 'kitti':
        classes = ['Car', 'Cyclist', 'Pedestrian']
        with open('/multiview/3d-count/Kitti/pp_result.pkl',"rb") as results_file:
            predictions = pkl.load(results_file)
        with open('/multiview/3d-count/Kitti/kitti_infos_val.pkl',"rb") as gt_file:
            gts = pkl.load(gt_file)
        thresholds = args.scores.split(",")
        for thresh in thresholds:
            thresh = float(thresh)
            prediction_count = []
            gt_count = []

            
            
            for k in tqdm(range(len(predictions))):
                mask = []
                for val in gts[k]['annos']['name']:
                    if val in classes:
                        mask.append(True)
                    else:
                        mask.append(False)
                gt_count.append(len(gts[k]['annos']['name'][mask]))
                prediction_count.append(len(predictions[k]['score'][predictions[k]['score'] > float(thresh)]))
            mae = np.mean(abs(np.array(gt_count) - np.array(prediction_count)))
            mape = np.mean(np.abs((np.array(gt_count) - np.array(prediction_count)) / np.array(gt_count))) * 100
            mse = np.mean(np.square(np.array(gt_count) - np.array(prediction_count)))
            print("threshold: ", thresh)
            print("mae: ", mae)
            print("mape: ", mape)
            print("mse: ", mse)
            print()

        





if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--dataset', type=str, help='path to scene')
    parser.add_argument('--scores', type=str, help='list of scores to try')
    args = parser.parse_args()
    main(args)