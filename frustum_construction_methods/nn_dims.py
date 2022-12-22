import cv2
import torch
from torch import nn
import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from wandb_utils.wandb import WandB

class NNDims(object):
    def __init__(self, boxes=[], classes=['Car', 'Pedestrian', 'Cyclist'], image=None, plot_image=True):
        self.original_boxes = boxes
        self.net = Frustum_Dims()
        self.classes = classes
        self.image = image
        self.new_boxes = self.run_nn()
        if plot_image:
            self.plotted_image = self.plot_bounding_boxes()
    
    def run_nn(self):
        new_box_dict = {}
        for key in self.original_boxes.keys():
            box_list = self.get_nn_boxes(self.original_boxes[key])
            new_box_dict.update({
                key : box_list
            })
        return new_box_dict
    
    def get_nn_boxes(self, boxes):
        new_boxes = []
        for box in boxes:
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]
            x = int(left + ((right - left) / 2))
            y = int(top + ((bottom - top) / 2))
            w = float(right - left)
            h = float(bottom - top)
            out = self.net(torch.tensor([[w,h]])).squeeze(0)
            w = round(out[0].item(),2)
            h = round(out[1].item(),2)
            left = x - (w/2)
            top = y - (h/2)
            right = x + (w/2)
            bottom = y + (h/2)
            new_boxes.append([left,top,right,bottom])
        return new_boxes

    def plot_bounding_boxes(self):
        new_image = self.image.copy()
        # original_color = (0,255,255)
        new_color = (255,0,255)
        thickness = 3
        for key in self.new_boxes.keys():
            # print("key: ", key, " in new boxes")
            if self.new_boxes[key] != []:
                for j in range(len(self.new_boxes[key])):
                    left = int(self.new_boxes[key][j][0])
                    top = int(self.new_boxes[key][j][1])
                    right = int(self.new_boxes[key][j][2])
                    bottom = int(self.new_boxes[key][j][3])
                    # print("adding new box to image")
                    new_image = cv2.rectangle(new_image, (left,top), (right,bottom),new_color, thickness)
        return new_image

    

class Frustum_Dims(nn.Module):
    def __init__(self):
        super(Frustum_Dims, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2)
        )
    

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
    
    

if __name__=='__main__':

    name = 'test-nn_dims'
    run = WandB(
        project='Thesis',
        enabled=True,
        entity='nelsonsw5',
        name=name,
        job_type="visualize"
        )

    image_path = '/Users/stephennelson/Projects/Data/Kitti/training/image_2/000047.png'
    im = cv2.imread(image_path)

    label_path = '/Users/stephennelson/Projects/Data/Kitti/training/label_2/000047.txt'
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
        # print("box class: ", box_class)
        if box_class in dataset_classes:
            left = int(float(object.split(" ")[4]))
            top = int(float(object.split(" ")[5]))
            right = int(float(object.split(" ")[6]))
            bottom = int(float(object.split(" ")[7]))
            boxes[box_class].append([left,top,right,bottom])
    scan = NNDims(boxes=boxes, classes=dataset_classes, image=im)
    img_dict = run.get_img_log(scan.plotted_image)
    dict_list = [img_dict]
    log_dict = {}
    for d in dict_list:
        for k, v in d.items():
            log_dict[k] = v
    run.log(log_dict)

    