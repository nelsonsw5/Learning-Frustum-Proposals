import cv2
import torch
from torch import nn


class NN_Dims(object):
    def __init__(self, boxes):
        self.boxes = boxes

    
    
    

class LearnDims(nn.Module):
    def __init__(self):
        super(LearnDims, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2)
        )

    def forward(self, x):
        print(x.shape)
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        print("out: ", out)
        return out





if __name__=='__main__':
    print("test")
    input = torch.tensor([[50.0,70.0]])
    print("input: ", input)
    network = LearnDims()
    output = network(input)
    print("test finished")