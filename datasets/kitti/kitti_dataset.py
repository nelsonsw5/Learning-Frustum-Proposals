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
        frustum_class_dict = {}
        for key in boxes.keys():
            # print("Number of boxes for: ", key, len(boxes[key]))
            frustum_class_dict.update({key:[]})
            for box in range(len(boxes[key])):
                frustum_dict = {}
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
                # print(pc_in_box_fov)
                if len(pc_in_box_fov) == 0:
                    # print("true")
                    continue
                pc_in_box_fov = calib.project_rect_to_velo(pc_in_box_fov[:,0:3])
                frustum_dict.update({
                    'Frustum_ID':box,
                    'Points':pc_in_box_fov,
                    'Centroids':[],
                    'Dims': []
                })
                point_x_min = np.min(pc_in_box_fov[:,0])
                point_y_min = np.min(pc_in_box_fov[:,1])
                point_z_min = np.min(pc_in_box_fov[:,2])

                point_x_max = np.max(pc_in_box_fov[:,0])
                point_y_max = np.max(pc_in_box_fov[:,1])
                point_z_max = np.max(pc_in_box_fov[:,2])

                for object in self.label:
                    if key != object.type:
                        continue
                    x = object.loc[0]
                    y = object.loc[1]
                    z = object.loc[2]
                    centroid = [x,y,z]
                    dims = [object.h,object.w,object.l]
                    h = object.h
                    w = object.w
                    l = object.l
                    centroid = np.array([centroid])
                    dims = np.array([dims])
                    centroid = np.array(self.calib.project_rect_to_velo(centroid))
                    h, w, l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    dims = np.concatenate([l,w,h],axis=1)
                    centroid[:, 2] += h[:, 0] / 2
                    x = centroid[0,0]
                    y = centroid[0,1]
                    z = centroid[0,2]

                    if (x > point_x_min) and (x < point_x_max):
                        
                        if (y > point_y_min) and (y < point_y_max):

                            if (z > point_z_min) and (z < point_z_max):
                                frustum_dict['Centroids'].append(centroid[0])
                                frustum_dict['Dims'].append(dims[0])
                if len(frustum_dict['Centroids']) == 0:
                    continue
                else:
                    frustum_class_dict[key].append(frustum_dict)
        return frustum_class_dict

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
    


        