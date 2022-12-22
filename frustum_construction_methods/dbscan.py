import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from wandb_utils.wandb import WandB
 
 



class DBScan(object):
    def __init__(self, boxes=[], epsilon=100, classes=['Car', 'Pedestrian', 'Cyclist'], image=None, plot_clusters=False, plot_image=True):
        self.original_boxes = boxes
        self.epsilon = epsilon
        self.classes = classes
        self.image = image
        self.plot_clusters = plot_clusters
        self.new_boxes = self.run_dbscan()
        if plot_image:
            self.plotted_image = self.plot_bounding_boxes()
        
    def plot_bounding_boxes(self):
        new_image = self.image.copy()
        new_color = (255,0,255)
        thickness = 3
        for key in self.new_boxes.keys():
            if self.new_boxes[key] != []:
                for j in range(len(self.new_boxes[key])):
                    left = int(self.new_boxes[key][j][0])
                    top = int(self.new_boxes[key][j][1])
                    right = int(self.new_boxes[key][j][2])
                    bottom = int(self.new_boxes[key][j][3])
                    new_image = cv2.rectangle(new_image, (left,top), (right,bottom),new_color, thickness)
        return new_image

    def get_clusters(self, boxes):
        centroids = []
        X = []
        Y = []
        for box in boxes:
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]
            x = int(left + ((right - left) / 2))
            y = int(top + ((bottom - top) / 2))
            centroids.append([x,y])
            X.append(x)
            Y.append(y)
        if self.plot_clusters:
            plt.scatter(X, Y, c ="blue")
            plt.savefig('plot_clusters.png')
        centroids = np.array(centroids)
        cluster_dict = {}
        clusters = DBSCAN(eps=self.epsilon, min_samples=2).fit_predict(centroids)
        for cluster_key in np.unique(clusters):
            cluster_dict.update({
                cluster_key: []
            })
        for cluster_id in range(len(clusters)):
            cluster_dict[clusters[cluster_id]].append(boxes[cluster_id]) 
        return cluster_dict

    def merge_boxes(self, boxes):
        box_dims = np.array(boxes)
        minx = box_dims.min(axis=0)[0]
        miny = box_dims.min(axis=0)[1]
        maxx = box_dims.max(axis=0)[2]
        maxy = box_dims.max(axis=0)[3]
        new_box = [minx, miny, maxx, maxy]
        return new_box

    def get_cluster_boxes(self, boxes):
        new_boxes = []
        clusters = self.get_clusters(boxes)
        for cluster in clusters.keys():
            if cluster == -1:
                for box in clusters[cluster]:
                    new_boxes.append(box)
            else:
                new_boxes.append(self.merge_boxes(clusters[cluster]))
        return new_boxes



    def run_dbscan(self):
        new_box_dict = {}
        for key in self.original_boxes.keys():
            if self.original_boxes[key] != []:
                box_list = self.get_cluster_boxes(self.original_boxes[key])
            else:
                box_list = []
            new_box_dict.update({
                key : box_list
            })
        return new_box_dict
        


if __name__=='__main__':

    name = 'test-dbscan'
    run = WandB(
        project='Thesis',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="visualize"
        )

    image_path = '/Users/stephennelson/Projects/Data/Kitti/training/image_2/000142.png'
    im = cv2.imread(image_path)

    label_path = '/Users/stephennelson/Projects/Data/Kitti/training/label_2/000142.txt'
    with open(label_path) as label_file:
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
    scan = DBScan(boxes=boxes, epsilon=100, classes=dataset_classes, image=im)
    img_dict = run.get_img_log(scan.plotted_image)
    dict_list = [img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)
    




