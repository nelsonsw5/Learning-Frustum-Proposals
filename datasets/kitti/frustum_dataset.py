import cv2
import os
import json
import numpy as np
from torch.utils.data import Dataset
import pdb


class FrustumDataset(Dataset):
    def __init__(self, data_path='/Users/stephennelson/Projects/Data/Frustum_kitti', test=False):
    
        """
        Args:
            data_path: path to where data lives
            test: true or false if evaluating
        """
        self.test = test
        self.point_path = os.path.join(data_path,'points')
        self.image_path = os.path.join(data_path,'image')
        self.label_path = os.path.join(data_path,'label')
        self.scene_list = self.get_scene_ids()
        self.classes = ['Car','Pedestrian','Cyclist']
        self.n_geo_types = len(self.classes)
        self.dataset_id = 'Kitti'

    def __getitem__(self, idx):
        filename = self.scene_list[idx]
        point_cloud_path = os.path.join(self.point_path,str(filename+'.bin'))
        frustum_label_path = os.path.join(self.label_path,str(filename+'.json'))
        points = self.get_points(point_cloud_path)
        label = self.get_frustum_count(frustum_label_path)
        
        return points, label
    
    def __len__(self):
        return len(self.scene_list)

    def get_points(self, file_path):
        cloud = np.fromfile(file_path, dtype=np.float32)
        cloud = cloud.reshape((-1, 3))
        return cloud

    def get_scene_ids(self):
        scene_itr = 0
        id_dict = {}
        directory = '/Users/stephennelson/Projects/Data/Frustum_kitti/label/'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                scene = filename.split(".")[0:-1]
                id_dict.update({scene_itr:scene[0]})
        return id_dict
    
    def get_frustum_count(self, filepath):
        with open(filepath,'r') as file:
            data = json.load(file)
        count = data['frustum_count']
        
        return count
    def get_n_geo_types(self):
        return self.n_geo_types

    def get_wandb_plot(self, wandb_run, batch, y, y_hat=None, eval_dict=None):
        

            log_dict = {}
            for d in dict_list:
                for k, v in d.items():
                    log_dict[k] = v

            wandb_run.log(log_dict)

if __name__ == '__main__':
    dataset = FrustumDataset()
    pdb.set_trace()
    print(dataset.__getitem__(0))