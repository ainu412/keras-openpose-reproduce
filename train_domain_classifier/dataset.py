import os
import cv2
import random
import math
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import *
import sys
import torchvision.transforms.functional as TF
import torch


def read_image(path):
    return Image.open(path)

class Dataset(data.Dataset):

    def __init__(self, path='../dataset/train2014_classifier/'):  # or '../dataset/val2014_classifier/'
        super(Dataset, self).__init__()
        self.path = path
        self.data = os.listdir(self.path)
        self.preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize([0.485], [0.229])
        ])


    def __getitem__(self, idx):
        img_file_name = self.data[idx]

        img = read_image(self.path + img_file_name).convert('RGB')
        # print('img size', img.size)
        img = self.preprocess(img)

        if img_file_name.endswith('_motion_blur.jpg'):
            label = torch.tensor([0, 0, 1])
        elif img_file_name.endswith('_dark.jpg'):
            label = torch.tensor([0, 1, 0])
        else:
            label = torch.tensor([1, 0, 0])

        # print('img_file_name', img_file_name, 'label', label)
        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_set = Dataset()
    for i in range(100):
        train_set[i]
