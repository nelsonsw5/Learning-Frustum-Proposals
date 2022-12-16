import cv2


class DBScan(object):
    def __init__(self, boxes, epsilon=.5, classes=['Car', 'Pedestrian', 'Cyclist']):
        self.boxes = boxes

        self.centroids = self.get_centroids(boxes)
        self.epsilon = epsilon
        self.classes = classes
        self.clusters = []

    def get_centroids(self,boxes):
        print("boxes: ", boxes)
        centroids = 'need to implement'
        print("getting centroids: ", centroids)

        return centroids


if __name__=='__main__':

    image_path = '/multiview/3d-count/Kitti/training/image_2/000072.png'
    im = cv2.imread(image_path)

    label_path = '/multiview/3d-count/Kitti/training/label_2/000072.txt'
    with open(label_path) as label_file:
        labels = label_file.readlines()
    dataset_classes = ['Car', 'Pedestrian', 'Cyclist']
    classes_in_scene = []
    boxes = []
    for object in labels:
        if object.split(" ")[0] in dataset_classes:
            classes_in_scene.append(object.split(" ")[0])
            left = int(float(object.split(" ")[4]))
            top = int(float(object.split(" ")[5]))
            right = int(float(object.split(" ")[6]))
            bottom = int(float(object.split(" ")[7]))
            boxes.append([left, top, right, bottom])
    scan = DBScan(boxes)




