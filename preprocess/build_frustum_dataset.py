import cv2
import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from wandb_utils.wandb import WandB
from frustum_construction_methods.iou import IoU
from frustum_construction_methods.dbscan import DBScan
from datasets.kitti.kitti_dataset import KittiDataset, KittiScene
from transforms import KittiPoints2Wandb, KittiLabel2Wandb
import torch
import pdb

DATA_PATH =  '/Users/stephennelson/Projects/Data/'

def create_new_folder(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return filepath



def build_frustum_scene(kitti_scene, save_path, frustum_method):
    
    boxes = kitti_scene.boxes
    if frustum_method == 'iou':
        iou = IoU(boxes=boxes, classes=kitti_scene.boxes, image=kitti_scene.image)
        image = iou.plotted_image
        new_boxes = iou.new_boxes

    elif frustum_method == 'dbscan':
        dbscan = DBScan(boxes=boxes, classes=kitti_scene.boxes, image=kitti_scene.image)
        image = dbscan.plotted_image
        new_boxes = dbscan.new_boxes
    else:
        new_boxes = boxes
        
    frustums, frustum_keys, pc, centroids, dims, cloud_dims, cloud_centroids = kitti_scene.build_frustums(new_boxes)

    return frustums, frustum_keys, image, pc, centroids, dims, cloud_dims, cloud_centroids


def create_frustum_data(dataset, new_data, frustum_method, viz):
    images = os.path.join(dataset.data_path,'image_2')
    for scene in os.listdir(images):
        # checking if it is a file
        if os.path.isfile(os.path.join(images,scene)):
            scene = scene[0:-4]
            scene = '000142'
            kitti_scene = KittiScene(scene_id=scene, dataset=dataset)

            frustums, frustum_labels, image, pc, centroids, dims, cloud_dims, cloud_centroids  = build_frustum_scene(kitti_scene,new_data,frustum_method)

            if viz:
                name = 'Viz-Frustum-Dataset'
                run = WandB(
                    project='Thesis',
                    enabled=True,
                    entity='nelsonsw5',
                    name=name,
                    job_type="Viz"
                    )
                dict_list = []
                point_swap = KittiPoints2Wandb()
                label_swap = KittiLabel2Wandb()


                cloud = point_swap(torch.tensor(pc[:,0:3]))
                cloud_centroids = label_swap(torch.tensor(cloud_centroids))
                cloud_dims = label_swap(torch.tensor(cloud_dims))
                cloud_boxes = run.get_bb_dict(cloud_centroids, cloud_dims, labels=None, colors=None)

                centroids = point_swap(torch.tensor(centroids))
                dims = point_swap(torch.tensor(dims))

                dict_list.append(run.get_point_cloud_log(cloud, boxes=cloud_boxes, key='Whole Cloud'))
                for pc in range(len(frustums)):
                    frustum = point_swap(torch.tensor(frustums[pc][:,0:3]))
                    boxes = run.get_bb_dict(centroids, dims, labels=None, colors=None)
                    dict_list.append(run.get_point_cloud_log(frustum, boxes=boxes, key=str(frustum_labels[pc])))
                dict_list.append(run.get_img_log(image, key=frustum_method))
                dict_list.append(run.get_img_log(kitti_scene.get_plotted_gt(), key='Original'))
                log_dict = {}
                for d in dict_list:
                    for k, v in d.items():
                        log_dict[k] = v
                run.log(log_dict)
            break
    return







def main(args):
    if args.dataset == 'Kitti':
        OG_Directory = os.path.join(DATA_PATH,args.dataset)
        dataset = KittiDataset(OG_Directory)
        New_Directory = os.path.join(DATA_PATH,'Frustum_Kitti')
        create_frustum_data(dataset,New_Directory, args.frustum_method, args.viz)
    
    elif args.dataset == 'SunRGBD':
        print("need to implement method")

    return




def parse_args():
    parser = argparse.ArgumentParser('Scene')
    parser.add_argument('--dataset', type=str, default="Kitti", help='kitti, sunrgbd')
    parser.add_argument('--frustum_method', type=str, default="dbscan", help='dbscan, iou')
    parser.add_argument('--viz', type=bool, default=True, help='True, False')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

