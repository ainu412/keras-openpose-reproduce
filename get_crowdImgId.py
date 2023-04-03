import json
import os
import random
import shutil
import pandas as pd
import math
import cv2
import copy
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)

    sample = random.sample(pathDir, 1000)

    # open file in write mode
    with open(tarDir[:-1] + '.txt', 'w') as fp:
        for item in sample:
            # write each item on a new line
            fp.write("%s\n" % item)

    print(sample)

    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)

def get_crowded_imgid():
    ann_path = 'dataset/annotations/person_keypoints_val2014.json'
    with open(ann_path, 'r') as f:
        data = json.load(f)
    crowded_imgIds = []
    for dic in data['annotations']:
        if dic['iscrowd'] == 1:
            crowded_imgIds.append(dic['image_id'])

def half_resolution():
    copyFile('dataset/val2014/', 'dataset/val2014_random1k_resolution_original/')
    fileDir = 'dataset/val2014_random1k_resolution_original/'
    finalDir = 'dataset/val2014_random1k_resolution/'
    fileNames = os.listdir(fileDir)
    for name in fileNames:
        src = cv2.imread(fileDir + name, cv2.IMREAD_UNCHANGED)
        dst = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2))
        cv2.imwrite(finalDir + name, dst)

def half_resolution_dataset_annotation():
    # half resolution dataset annotations
    ann_path = 'dataset/annotations/person_keypoints_val2014.json'
    with open(ann_path, 'r') as f:
        data = json.load(f)
    half_data = copy.deepcopy(data)
    for dic_i, dic in enumerate(half_data['annotations']):
        for i in range(51):
            if i % 3 == 2:
                continue
            half_data['annotations'][dic_i]['keypoints'][i] /= 2
        half_data['annotations'][dic_i]['area'] /= 4
        if len(half_data['annotations'][dic_i]['segmentation']) == 1:
            half_data['annotations'][dic_i]['segmentation'][0] = \
                list(np.array(half_data['annotations'][dic_i]['segmentation'][0]) / 2)

    ann_half_path = 'dataset/annotations/person_keypoints_val2014_random1k_resolution.json'
    with open(ann_half_path, 'w') as final:
        json.dump(half_data, final, cls=NpEncoder)

def motion_blur_dataset():
    dir = 'dataset/val2014_random1k/'
    fileNames = os.listdir(dir)
    for name in fileNames:
        img = cv2.imread(dir + name)

        # Specify the kernel size.
        # The greater the size, the more the motion.
        kernel_size = 30

        # Create the vertical kernel.
        # kernel_v = np.zeros((kernel_size, kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.zeros((kernel_size, kernel_size))

        # Fill the middle row with ones.
        # kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

        # Normalize.
        # kernel_v /= kernel_size
        kernel_h /= kernel_size

        # Apply the vertical kernel.
        # vertical_mb = cv2.filter2D(img, -1, kernel_v)
        # Apply the horizontal kernel.
        horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        # Save the outputs.
        # cv2.imwrite('car_vertical.jpg', vertical_mb)
        cv2.imwrite('dataset/val2014_random1k_motion_blur/' + name, horizonal_mb)

if __name__ == '__main__':
    # val 2014的所有文件中随机选1000个存到另一个文件夹中去
    # copyFile('dataset/val2014/', 'dataset/val2014_random1k/')
    # df = pd.read_csv('eval/deform_rare_crowded_list.csv')
    # deform_id = set([int(v) for v in df['deformation'].values if not math.isnan(v)])
    # rare_id = set([int(v) for v in df['rare_novel_poses'].values if not math.isnan(v)])
    # crowd_back_id = set([int(v) for v in df['crowded_background'].values if not math.isnan(v)])


    # half_resolution_dataset_annotation()
    # motion_blur_dataset()
    dir = 'dataset/val2014_random1k/'
    fileNames = os.listdir(dir)
    # open file in write mode
    with open('dataset/val2014_random1k_motion_blur.txt', 'w') as fp:
        for item in fileNames:
            # write each item on a new line
            fp.write("%s\n" % int(item[13:-4]))