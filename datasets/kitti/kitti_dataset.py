import copy
import pickle
import cv2
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import kitti_util as util
from torch.utils.data import Dataset
from bound_box import ThreeDimBoundBox
import pdb


class KittiDataset(Dataset):
    def __init__(self, data_path='/multiview/3d-count/Kitti/', test=False):
    
        """
        Args:
            data_path: path to where data lives
            test: true or false if evaluating
        """
        self.test = test
        self.data_path = os.path.join(data_path,"training")
        self.id_files = os.path.join(data_path, "ImageSets")
        self.data_ids = self.load_dataset_ids()
        self.classes = ['Car', 'Pedestrian','Cyclist']
        self.n_geo_types = len(self.classes)
        self.dataset_id = 'Kitti'

    def load_dataset_ids(self):
        ids = []
        if self.test:
            path = os.path.join(self.id_files,"val.txt")
        else:
            path = os.path.join(self.id_files,"train.txt")
        
        with open(path) as txt_file:
            lines = txt_file.readlines()
        for id in lines:
            ids.append(id[0:-1])
        return ids

    def get_n_geo_types(self):

        return self.n_geo_types

    def __len__(self):

        return len(self.data_ids)

    def __getitem__(self, idx):
        input = {}
        input.update({})
        return input

class KittiScene(object):
    def __init__(self, scene_id, dataset):
        self.dataset = dataset
        self.scene_id = scene_id
        self.pc = self.get_point_cloud(scene_id)
        self.label = self.get_label(scene_id)
        self.calib = self.get_calib(scene_id)
        self.image = self.get_image(scene_id)
        self.boxes = self.get_scene_boxes()
        self.gt_plotted_image = self.get_plotted_gt()
        
    def get_point_cloud(self, idx):
        cloud_path = os.path.join(self.dataset.data_path,'velodyne',str(idx + '.bin'))
        pc = util.load_velo_scan(cloud_path)
        return pc
    
    def get_label(self, idx):
        label_path = os.path.join(self.dataset.data_path,'label_2',str(idx + '.txt'))
        label = util.read_label(label_path)
        return label

    def get_calib(self, idx):
        calib_path = os.path.join(self.dataset.data_path,'calib',str(idx + '.txt'))
        calib = util.Calibration(calib_path)
        return calib

    def get_image(self, idx):
        image_path = os.path.join(self.dataset.data_path,'image_2',str(idx + '.png'))
        image = util.load_image(image_path)
        return image

    def get_scene_boxes(self):
        boxes = {
            'Car':[],
            'Pedestrian':[],
            'Cyclist':[]
        }
        for object in self.label:
            box_class = object.type
            if box_class in self.dataset.classes:
                boxes[box_class].append(object.box2d)
        return boxes
    
    def build_frustums(self, boxes):
        calib = self.calib # 3 by 4 matrix
        objects = self.label
        pc_velo = self.pc
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = self.image
        frustum_pcs = []
        frustum_scene_label = []
        img_height, img_width, img_channel = img.shape
        # pc_image_coord = point cloud points in 2D
        # img_fov_inds = True/False for each point in cloud in image field of view
        _, pc_image_coord, img_fov_inds = util.get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

        for key in boxes.keys():
            for box in range(len(boxes[key])):
                frustum_label_dict = {'Frustum_class':key}
                #xy min and xy max for bounding boxes
                xmin,ymin,xmax,ymax = boxes[key][box][0],boxes[key][box][1],boxes[key][box][2],boxes[key][box][3]
                #box_fov_inds = points in 2d that fall inside bounding box coordinates
                box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                    (pc_image_coord[:,0]>=xmin) & \
                    (pc_image_coord[:,1]<ymax) & \
                    (pc_image_coord[:,1]>=ymin)
                #box_fov_inds = points in 2d box and that are in the image field of view
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:] #(1607, 4)
                pc_in_box_fov = calib.project_rect_to_velo(pc_in_box_fov[:,0:3])
                
                for object in self.label:
                    x = object.loc[0]
                    y = object.loc[1]
                    z = object.loc[2]
                    
                    # 3D BOX: Get pts velo in 3d box
                    box3d_pts_2d, box3d_pts_3d = util.compute_box_3d(obj, calib.P) #(8, 2)(8, 3)
                    boxes3D.append(box3d_pts_3d)
                frustum_pcs.append(pc_in_box_fov)
                frustum_scene_label.append(str(key + " " + str(box)))
                
                

        centroids, dims = util.get_box_params(boxes3D)
        h, w, l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        dims = np.concatenate([l,w,h],axis=1)
        centroids[:, 0] += l[:, 0] / 2
        return input_list, frustum_labels, pc_velo, centroids, dims, cloud_dims, cloud_centroids

    def get_plotted_gt(self):
        new_image = self.image.copy()
        new_color = (0,255,255)
        thickness = 3
        text_thickness = 2
        text_size = .5
        for key in self.boxes.keys():
            if self.boxes[key] != []:
                for j in range(len(self.boxes[key])):
                    left = int(self.boxes[key][j][0])
                    top = int(self.boxes[key][j][1])
                    right = int(self.boxes[key][j][2])
                    bottom = int(self.boxes[key][j][3])
                    new_image = cv2.rectangle(new_image, (left,top), (right,bottom),new_color, thickness)
                    # new_image = cv2.putText(new_image, key + " Box: " + str(j), (left-10,top-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, new_color, text_thickness)
        return new_image


        
    
    


if __name__ == '__main__':
    print("Kitti Dataset Class")
    


        