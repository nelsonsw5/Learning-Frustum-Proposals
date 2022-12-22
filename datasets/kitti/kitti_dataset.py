import copy
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import kitti_util as utils
from torch.utils.data import Dataset
import pdb


class KittiDataset(Dataset):
    def __init__(self, data_path='/multiview/3d-count/Kitti/', test=False):
    
        """
        Args:
            data_path: path to where data lives
            test: true or false if evaluating
        """
        self.data_path = data_path + "training/"
        self.id_files = data_path + "ImageSets/"
        self.data_ids = self.load_dataset_ids(self.id_files, test)
        self.classes = ['Car', 'Pedestrian','Cyclist']
        self.n_geo_types = len(self.classes)
        self.max_val = self.get_max_val()

    def load_dataset_ids(self, data_path, test):
        ids = []
        if test:
            path = data_path + "val.txt"
        else:
            path = data_path + "train.txt"
        with open(path) as txt_file:
            lines = txt_file.readlines()
        for id in lines:
            ids.append(id[0:-1])
        return ids

    def get_n_geo_types(self):

        return self.n_geo_types

    def get_point_cloud(self, idx):
        cloud_path = self.data_path + 'velodyne/' + idx + '.png'
        pc = util.load_velo_scan(cloud_path)
        return pc
    
    def get_label(self, idx):
        label_path = self.data_path + 'label_2/' + idx + '.png'
        label = util.read_label(label_path)
        return label

    def get_calib(self, idx):
        calib_path = self.data_path + 'calib/' + idx + '.txt'
        calib = util.Calibration(calib_path)
        return calib

    
    def get_image(self, idx):
        image_path = self.data_path + 'image_2/' + idx + '.png'
        image = util.load_image(image_path)
        return image
    
    def __len__(self):

        return len(self.data_ids)

    def __getitem__(self, idx):
        input = {}
        pc = get_point_cloud(idx)
        label = get_label(idx)
        calib = get_calib(idx)
        image = get_image(idx)

        input.update({})

        return input
    def get_max_val(self):
        max_val = 15

        return max_val



        


if __name__ == '__main__':
    dataset = KittiDataset()
    print(dataset.__len__())
    


        