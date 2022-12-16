import os
import sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from wandb_utils.wandb import WandB



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
        with open('/multiview/3d-count/Kitti/training/label_2/'+ scene + '.txt') as label_file:
            lines = label_file.readlines()
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
        
    

    if args.dataset == "sunrgbd":
        print("need to implement this")
    

    

    pc_dict = run.get_point_cloud_log(pc)
    img_dict = run.get_img_log(im)
    dict_list = [pc_dict, img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)



def parse_args():
    parser = argparse.ArgumentParser('Scene')
    parser.add_argument('--scene', type=str, default="3", help='scene index number')
    parser.add_argument('--dataset', type=str, default="kitti", help='kitti, sunrgbd, coolers')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)