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
import json
import torch
import pdb
from tqdm import tqdm

DATA_PATH =  '/Users/stephennelson/Projects/Data/'

def create_new_folder(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return filepath

def write_frustum_data(frustum):
    centroids = []
    dims = []
    for centroid in frustum['Centroids']:
        centroids.append(list(centroid))
    for dim in frustum['Dims']:
        dims.append(list(dim))
    label_dict = {'centroids':centroids,'dimensions':dims,'frustum_count':len(frustum['Centroids'])}
    points = frustum['Points']
    # print("build_frustum_dataset 33")
    # print(points)
    return label_dict, points

def save_scene(frustum_data,scene_image, scene, save_path):
    # print("scene: ", scene)
    # print("save_path: ", save_path)
    image_path = os.path.join(save_path,'image')
    label_path = os.path.join(save_path,'label')
    points_path = os.path.join(save_path,'points')
    for key in frustum_data.keys():
        for scene_id in range(len(frustum_data[key])):
            scene_image_path = os.path.join(image_path,str(scene+'-'+key+"-"+str(scene_id)+'.png'))
            scene_label_path = os.path.join(label_path,str(scene+'-'+key+"-"+str(scene_id)+'.json'))
            scene_points_path = os.path.join(points_path,str(scene+'-'+key+"-"+str(scene_id)+'.npy'))
            # print("build_frustum_dataset 43")
            # print("scene_points_path: ", scene_points_path)
            cv2.imwrite(scene_image_path,scene_image)
            label, points = write_frustum_data(frustum_data[key][scene_id])
            label_json = json.dumps(label, indent=4)
            with open(scene_label_path, "w") as outfile:
                outfile.write(label_json)
            np.save(scene_points_path,points)
    return



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
        
    frustums_dict = kitti_scene.build_frustums(new_boxes)

    return frustums_dict, image

def plot_whole_scene(kitti_scene, swap):
    pc = swap(torch.tensor(kitti_scene.pc[:,0:3]))
    all_centroids = []
    all_dims = []
    for object in kitti_scene.label:
        x = object.loc[0]
        y = object.loc[1]
        z = object.loc[2]
        all_centroids.append([x,y,z])
        all_dims.append([object.h,object.w,object.l])
    all_centroids = np.array(all_centroids)
    all_dims = np.array(all_dims)
    all_centroids = np.array(kitti_scene.calib.project_rect_to_velo(all_centroids))
    h, w, l = all_dims[:, 0:1], all_dims[:, 1:2], all_dims[:, 2:3]
    all_dims = np.concatenate([l,w,h],axis=1)
    all_centroids[:, 2] += h[:, 0] / 2

    return pc, all_centroids, all_dims


def create_frustum_data(dataset, new_data, frustum_method, viz):
    images = os.path.join(dataset.data_path,'image_2')
    for scene in tqdm(os.listdir(images)):
        # checking if it is a file
        if os.path.isfile(os.path.join(images,scene)):
            scene = scene[0:-4]
            # scene = '000076'
            # print("scene: ", scene)
            kitti_scene = KittiScene(scene_id=scene, dataset=dataset)

            frustum_dict, image = build_frustum_scene(kitti_scene,new_data,frustum_method)
            if viz != 'none':
                name = 'Viz-Frustum-Dataset'
                run = WandB(
                    project='Frustum-CounNet',
                    enabled=True,
                    entity="prj-research",
                    name=name,
                    job_type="Viz_Dataset_Build"
                    )
                dict_list = []
                point_swap = KittiPoints2Wandb()
                if viz == 'all':
                    pc, all_centroids, all_dims = plot_whole_scene(kitti_scene,point_swap)
                    cloud_boxes = run.get_bb_dict(point_swap(torch.tensor(all_centroids)), point_swap(torch.tensor(all_dims)), labels=None, colors=None)
                    dict_list.append(run.get_point_cloud_log(pc, boxes=cloud_boxes, key='Whole Cloud'))
                for key in frustum_dict.keys():
                    for frustum in frustum_dict[key]:
                        points = point_swap(torch.tensor(frustum['Points']))
                        if len(frustum['Centroids']) == 0:
                            centroids = []
                            dims = []
                        else:
                            centroids = frustum['Centroids']
                            dims = frustum['Dims']
                            centroids = np.array(centroids)
                            dims = np.array(dims)
                        boxes = run.get_bb_dict(point_swap(torch.tensor(centroids)), point_swap(torch.tensor(dims)), labels=None, colors=None)
                        dict_list.append(run.get_point_cloud_log(points, boxes=boxes, key=str(key + "-" + str(frustum['Frustum_ID']))))
                    dict_list.append(run.get_img_log(image, key=frustum_method))
                    dict_list.append(run.get_img_log(kitti_scene.get_plotted_gt(), key='Original'))
                log_dict = {}
                for d in dict_list:
                    for k, v in d.items():
                        log_dict[k] = v
                run.log(log_dict)
            save_scene(frustum_dict,kitti_scene.image,scene,new_data)
            # break
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
    parser.add_argument('--viz', type=str, default="none", help='all, frustums, none')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

