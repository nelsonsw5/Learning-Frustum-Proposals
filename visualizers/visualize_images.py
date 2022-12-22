import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from frustum_construction_methods.dbscan import DBScan
from frustum_construction_methods.iou import IoU
from frustum_construction_methods.nn_dims import NNDims
from wandb_utils.wandb import WandB
import argparse

def main(args):
    scene = int(args.scene)
    name = 'viz method ' + args.scene + ' ' + args.dataset + ' ' + args.frustum_method
    run = WandB(
        project='Thesis',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="visualize"
        )
    
    if args.dataset == "kitti":
        scene = str(args.scene).zfill(6)
        im = cv2.imread('/Users/stephennelson/Projects/Data/Kitti/training/image_2/' + scene + '.png')
        with open('/Users/stephennelson/Projects/Data/Kitti/training/label_2/'+ scene + '.txt') as label_file:
            labels = label_file.readlines()
        dataset_classes = ['Car', 'Pedestrian', 'Cyclist']
        boxes = {
            'Car':[],
            'Pedestrian':[],
            'Cyclist':[]
        }
        for object in labels:
            box_class = object.split(" ")[0]
            if box_class in dataset_classes:
                left = int(float(object.split(" ")[4]))
                top = int(float(object.split(" ")[5]))
                right = int(float(object.split(" ")[6]))
                bottom = int(float(object.split(" ")[7]))
                boxes[box_class].append([left,top,right,bottom])
        images = []
        key_list = []
        if args.plot_og:
            image_og = im.copy()
            for key in boxes.keys():
                for box in boxes[key]:
                    image_og = cv2.rectangle(image_og, (box[0],box[1]), (box[2],box[3]), (255,0,255), 3)
            images.append(image_og)
            key_list.append("Original")
        if args.frustum_method == 'dbscan' or args.frustum_method == 'all':
            # print("running dbscan")
            clusters = DBScan(boxes=boxes, epsilon=100, classes=dataset_classes, image=im, plot_clusters=True, plot_image=True)
            db_image = clusters.plotted_image
            images.append(db_image)
            key_list.append('DBScan')

        if args.frustum_method == 'iou' or args.frustum_method == 'all':
            # print("running iou")
            merged = IoU(boxes=boxes, classes=dataset_classes, image=im, plot_image=True)
            iou_image = merged.plotted_image
            images.append(iou_image)
            key_list.append('Iou')

        if args.frustum_method == 'nn' or args.frustum_method == 'all':
            prenet = NNDims(boxes=boxes, classes=dataset_classes, image=im, plot_image=True)
            nn_image = prenet.plotted_image
            images.append(nn_image)
            key_list.append('NN')

    dict_list = []
    for i in range(len(images)):
        dict_list.append(run.get_img_log(images[i], key_list[i]))
    log_dict = {}
    # print("dict_list: ", dict_list)
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)






def parse_args():
    parser = argparse.ArgumentParser('Scene')
    parser.add_argument('--scene', type=str, default="7", help='scene index number')
    parser.add_argument('--dataset', type=str, default="kitti", help='kitti, sunrgbd, coolers')
    parser.add_argument('--frustum_method', type=str, default="dbscan", help='dbscan, iou, nn, all')
    parser.add_argument('--plot_og', type=bool, default=True, help='Boolean')
    return parser.parse_args()







if __name__=='__main__':
    args = parse_args()
    main(args)


    