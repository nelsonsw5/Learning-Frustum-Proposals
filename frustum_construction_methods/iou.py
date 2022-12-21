import cv2
import pdb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from wandb_utils.wandb import WandB

class IoU(object):
    def __init__(self, boxes=[], classes=['Car', 'Pedestrian', 'Cyclist'], image=None):
        self.original_boxes = boxes
        self.new_boxes = self.run_iou()
        self.classes = classes
        if len(image) != 0:
            self.image = image
    
    def plot_bounding_boxes(self):
        run = WandB(
            project='Thesis',
            enabled=True,
            entity='nelsonsw5',
            name='IoU',
            job_type="Debugging"
        )
        new_image = self.image.copy()
        # original_color = (0,255,255)
        new_color = (255,0,255)
        thickness = 3
        # for key in self.original_boxes.keys():
        #     print("key: ", key, " in orignal boxes")
        #     if self.original_boxes[key] != []:
        #         for i in range(len(self.original_boxes[key])):
        #             left = int(self.original_boxes[key][i][0])
        #             top = int(self.original_boxes[key][i][1])
        #             right = int(self.original_boxes[key][i][2])
        #             bottom = int(self.original_boxes[key][i][3])
        #             print("adding old box to image")
        #             new_image = cv2.rectangle(new_image, (left,top), (right,bottom),original_color, thickness)

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
        
        img_dict = run.get_img_log(new_image)
        dict_list = [img_dict]
        log_dict = {}
        for d in dict_list:
            for k, v in d.items():
                log_dict[k] = v
        run.log(log_dict)

        return
    
    def get_IoU_boxes(self, boxes):
        new_boxes = []
        # print("all boxes: ", boxes)
        # print()
        for box in boxes:
            # print("box for evaluation: ", box)
            # print("new_boxes before starting comparison loop: ", new_boxes)
            # print()
            added = False
            for tmpbox in new_boxes:
                # print("comparing eval box to: ", tmpbox)
                # print("new_boxes at start of comparison: ", new_boxes)
                iou, new_box = self.intersection_over_union(box, tmpbox)
                # print("iou: ", iou)
                if iou > 0:
                    added = True
                    # print("removing: ", tmpbox)
                    new_boxes.remove(tmpbox)
                    # print("new_boxes after removal: ", new_boxes)
                    # print("adding: ", new_box)
                    new_boxes.append(new_box)
                    # print("new_boxes after adding: ", new_boxes)
                    # print("BREAKING")
                    # print()
                    break
            if added == False:
                new_boxes.append(box)
                # print("added box to new_boxes because did not IOU: ", new_boxes)
                # print()

        return new_boxes

    def run_iou(self):
        new_box_dict = {}
        for key in self.original_boxes.keys():
            # print()
            # print("running iou alg on: ", key)
            # print()
            box_list = self.get_IoU_boxes(self.original_boxes[key])
            new_box_dict.update({
                key : box_list
            })
        return new_box_dict

    def intersection_over_union(self, box1, box2):
        # calculate the intersection area
        inter_rect_x1 = max(box1[0], box2[0])
        inter_rect_y1 = max(box1[1], box2[1])
        inter_rect_x2 = min(box1[2], box2[2])
        inter_rect_y2 = min(box1[3], box2[3])

        # intersection area
        inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
        
        # union area
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        # Calculate union area
        union_rect_x1 = min(box1[0], box2[0])
        union_rect_y1 = min(box1[1], box2[1])
        union_rect_x2 = max(box1[2], box2[2])
        union_rect_y2 = max(box1[3], box2[3])

        if iou > 0:
            return iou, (union_rect_x1, union_rect_y1, union_rect_x2, union_rect_y2)
        else:
            return iou, (0,0,0,0)

        


if __name__=='__main__':

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
        if box_class in dataset_classes:
            left = int(float(object.split(" ")[4]))
            top = int(float(object.split(" ")[5]))
            right = int(float(object.split(" ")[6]))
            bottom = int(float(object.split(" ")[7]))
            boxes[box_class].append([left,top,right,bottom])
    scan = IoU(boxes=boxes, classes=dataset_classes, image=im)
    scan.plot_bounding_boxes()




