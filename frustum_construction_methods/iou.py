import cv2


class IoU(object):
    def __init__(self, boxes=[], classes=['Car', 'Pedestrian', 'Cyclist'], image=None):
        self.original_boxes = boxes
        self.new_boxes = self.get_IoU_boxes()
        self.classes = classes
        if len(image) != 0:
            self.image = image
    
    def plot_bounding_boxes(self):
        new_image = self.image.copy()
        for box in self.original_boxes:
            pdb.set_trace()
            box
        return
    
    def get_IoU_boxes(self):
        new_boxes = []


        return new_boxes

    def intersection_over_union(box1, box2):
        if box_format == 'corners':
        
        # calculate the intersection area
        inter_rect_x1 = max(box1_x1, box2_x1)
        inter_rect_y1 = max(box1_y1, box2_y1)
        inter_rect_x2 = min(box1_x2, box2_x2)
        inter_rect_y2 = min(box1_y2, box2_y2)

        # intersection area
        inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
        
        # union area
        box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
        box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        # Calculate union area
        union_rect_x1 = min(box1_x1, box2_x1)
        union_rect_y1 = min(box1_y1, box2_y1)
        union_rect_x2 = max(box1_x2, box2_x2)
        union_rect_y2 = max(box1_y2, box2_y2)

        if iou > 0:
            return iou, (union_rect_x1, union_rect_y1, union_rect_x2, union_rect_y2)
        else:
            return iou, (0,0,0,0)

        


if __name__=='__main__':

    image_path = '/multiview/3d-count/Kitti/training/image_2/000072.png'
    im = cv2.imread(image_path)

    label_path = '/multiview/3d-count/Kitti/training/label_2/000072.txt'
    with open(label_path) as label_file:
        labels = label_file.readlines()
    dataset_classes = ['Car', 'Pedestrian', 'Cyclist']
    boxes = []
    for object in labels:
        box_dict = {}
        box_class = object.split(" ")[0]
        if box_class in dataset_classes:
            left = int(float(object.split(" ")[4]))
            top = int(float(object.split(" ")[5]))
            right = int(float(object.split(" ")[6]))
            bottom = int(float(object.split(" ")[7]))

            box_dict.update({
                'box_class' : box_class,
                'box' : [left, top, right, bottom]
            })
            boxes.append(box_dict)
    scan = IoU(boxes=boxes, classes=dataset_classes, image=im)
    scan.plot_bounding_boxes()




