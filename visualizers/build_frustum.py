import os
import sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
from wandb_utils.wandb import WandB
import pdb
from datasets.kitti.kitti_object import *
from datasets.kitti import kitti_util as util
from tqdm import tqdm
from transforms import NormalizeShift, NegZForward2YForward, MirrorX, ShufflePoints, KittiPoints2Wandb
import torch


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds




def main(args):
    name = 'test-viz scene ' + args.scene + ' ' + args.dataset
    run = WandB(
        project='Thesis',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="Viz-Test"
        )
    
    if args.dataset == "kitti":
        scene = str(args.scene).zfill(6)
        dataset = kitti_object()
        
        im = cv2.imread('/multiview/3d-count/Kitti/training/image_2/' + scene + '.png')
        point_cloud = '/multiview/3d-count/Kitti/training/velodyne/' + scene + ".bin"
        with open(point_cloud, "rb") as f:
            pc = np.fromfile(f, dtype=np.float32).reshape((-1,4))
        color = (0,255,255)
        thickness = 3

        kitti_classes = ['Car','Cyclist','Pedestrian']
        scene_detections = []
        for object in lines:
            if object.split(" ")[0] in kitti_classes:
                scene_detections.append(object[0:-1])
        for i in range(len(scene_detections)):
            detection = scene_detections[i].split(" ")
            left = int(float(detection[4]))
            top = int(float(detection[5]))
            right = int(float(detection[6]))
            bottom = int(float(detection[7]))
            im = cv2.rectangle(im, (left,top), (right,bottom), color, thickness)
        
        id_list = [] # int number
        box2d_list = [] # [xmin,ymin,xmax,ymax]
        box3d_list = [] # (8,3) array in rect camera coord
        input_list = [] # channel number = 4, xyz,intensity in rect camera coord
        label_list = [] # 1 for roi object, 0 for clutter
        type_list = [] # string e.g. Car
        heading_list = [] # ry (along y-axis in rect camera coord) radius of
        # (cont.) clockwise angle from positive x axis in velo coord.
        box3d_size_list = [] # array of l,w,h
        frustum_angle_list = [] # angle of 2d box center from pos x-axis
        calib_list = [] # calibration matrix 3x4 for fconvnet
        image_filename_list = [] # for fusion
        input_2d_list = []

        pos_cnt = 0
        all_cnt = 0

        calib = dataset.get_calibration(int(args.scene)) # 3 by 4 matrix
        objects = dataset.get_label_objects(int(args.scene))
        pc_velo = dataset.get_lidar(int(args.scene))
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(int(args.scene))
        img_height, img_width, img_channel = img.shape


        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
        
        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in kitti_classes:
                continue
            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            # Augment data by box2d perturbation
            xmin,ymin,xmax,ymax = box2d
            box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                (pc_image_coord[:,0]>=xmin) & \
                (pc_image_coord[:,1]<ymax) & \
                (pc_image_coord[:,1]>=ymin)
            box_fov_inds = box_fov_inds & img_fov_inds
            pc_in_box_fov = pc_rect[box_fov_inds,:] #(1607, 4)
            input_list.append(pc_in_box_fov)

    swap = KittiPoints2Wandb()
    frustum = swap(torch.tensor(input_list[0][:,0:3]))
    pc_dict = run.get_point_cloud_log(frustum,boxes=None)
    img_dict = run.get_img_log(im)
    dict_list = [pc_dict, img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)

def parse_args():
    parser = argparse.ArgumentParser('Scene')
    parser.add_argument('--scene', type=str, default="7", help='scene index number')
    parser.add_argument('--dataset', type=str, default="kitti", help='kitti, sunrgbd, coolers')
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    main(args)

