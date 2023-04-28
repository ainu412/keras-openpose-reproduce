import json
import os
import random
import shutil
import pandas as pd
import math
import cv2
import copy
import numpy as np
from PIL import Image, ImageEnhance
import argparse


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def copyFile_sample1k(fileDir, tarDir):
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


def copyFile(dir, tar_dir):
    fileNames = os.listdir(dir)
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    for name in fileNames:
        shutil.copyfile(dir + name, tar_dir + name)


def get_crowded_imgid():
    ann_path = 'dataset/annotations/person_keypoints_val2014.json'
    with open(ann_path, 'r') as f:
        data = json.load(f)
    crowded_imgIds = []
    for dic in data['annotations']:
        if dic['iscrowd'] == 1:
            crowded_imgIds.append(dic['image_id'])


def half_resolution():
    fileDir = 'dataset/val2014_random1k/'
    finalDir = 'dataset/val2014_random1k_resolution/'
    if not os.path.exists(finalDir):
        os.mkdir(finalDir)
    fileNames = os.listdir(fileDir)
    for name in fileNames:
        src = cv2.imread(fileDir + name, cv2.IMREAD_UNCHANGED)
        dst = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2))
        cv2.imwrite(finalDir + name, dst)


def half_resolution_dataset_annotation():
    # half resolution dataset annotations
    ann_path = 'dataset/annotations/person_keypoints_val2014_random1k.json'
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
    dir = 'dataset/frames/'
    fileNames = os.listdir(dir)
    tar_dir = 'dataset/frames_motion_blur/'
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
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
        cv2.imwrite(tar_dir + name[:-4] + '_motion_blur' + '.png', horizonal_mb)


def dark_dataset():
    dir = 'dataset/frames/'
    fileNames = os.listdir(dir)
    tar_dir = 'dataset/frames_dark/'
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    for name in fileNames:
        # read the image
        img = Image.open(dir + name)

        # image brightness enhancer
        enhancer = ImageEnhance.Brightness(img)

        factor = 0.5  # darkens the image
        img_output = enhancer.enhance(factor)
        img_output.save(tar_dir + name[:-4] + '_dark' + '.jpg')


def compute_transl_rotate_disagreement(data_type):
    def euclidean_dist_2d(x1, x2, y1, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    prefix = 'person_keypoints'
    ann_path = ('dataset/annotations/%s_%s.json' % (prefix, data_type))

    with open(ann_path, 'r') as f:
        ann = json.load(f)['annotations']

    dark_path = 'eval/results/%s-dark-epoch100-open-pose-single-scale-1000/' \
                '%s-dark-epoch100-1000_result_matchGtId.json' \
                % (data_type, data_type)

    mb_path = 'eval/results/%s-motion_blur-epoch100-open-pose-single-scale-1000/' \
              '%s-motion_blur-epoch100-1000_result_matchGtId.json' \
              % (data_type, data_type)
    with open(mb_path, 'r') as f:
        motion_blur_json = json.load(f)
    with open(dark_path, 'r') as f:
        dark_json = json.load(f)

    ours_path = 'eval/results/%s-ours-epoch100-open-pose-single-scale-1000/' \
                '%s-ours-epoch100-1000_result_matchGtId.json' \
                % (data_type, data_type)
    with open(ours_path, 'r') as f:
        ours_json = json.load(f)

    # sort according to img_id
    motion_blur_json = sorted(motion_blur_json, key=lambda d: d['image_id'])
    dark_json = sorted(dark_json, key=lambda d: d['image_id'])
    ours_json = sorted(ours_json, key=lambda d: d['image_id'])

    num_body_parts = 17
    body_part_translational_dis = [[] for i in range(num_body_parts)]
    body_part_rotational_dis = [[] for i in range(num_body_parts)]
    for i in range(num_body_parts):
        if len(motion_blur_json) >= len(dark_json) >= len(ours_json) \
                or len(motion_blur_json) >= len(ours_json) >= len(dark_json):
            longest_json = motion_blur_json
            json1 = dark_json
            json2 = ours_json
        elif len(dark_json) >= len(motion_blur_json) >= len(ours_json) \
                or len(dark_json) >= len(ours_json) >= len(motion_blur_json):
            longest_json = dark_json
            json1 = motion_blur_json
            json2 = ours_json
        else:
            longest_json = ours_json
            json1 = motion_blur_json
            json2 = dark_json
        for dic in longest_json:
            # search dic with same gtid
            if dic['gtId'] == 0:
                continue
            # res = list(filter(lambda d: d['gtId'] == dic['gtId'] , json2))
            # print('res', len(res[0]['body_link_list']))
            # res = list(filter(lambda d: d['gtId'] == dic['gtId'] , json1))
            # print('res', len(res[0]['body_link_list']) if res else "")

            res = list(filter(lambda d: d['gtId'] == dic['gtId'] and d['body_link_list'][i * 3 + 0] != [0, 0], json1)) + \
                  list(filter(lambda d: d['gtId'] == dic['gtId'] and d['body_link_list'][i * 3 + 0] != [0, 0], json2))

            gt = [a for a in ann if a['id'] == dic['gtId']][0]

            if gt['keypoints'][-2] == 0 or gt['keypoints'][-3] == 0 \
                    or gt['keypoints'][-5] == 0 or gt['keypoints'][-6] == 0:
                continue
            head_bone_link_length = euclidean_dist_2d(gt['keypoints'][-2], gt['keypoints'][-5],
                                                      gt['keypoints'][-3], gt['keypoints'][-6])
            # print('head_bone_link_length', head_bone_link_length)

            if not res:
                continue

            x1 = dic['body_link_list'][i * 3 + 0]
            y1 = dic['body_link_list'][i * 3 + 1]

            x2 = res[0]['body_link_list'][i * 3 + 0]
            y2 = res[0]['body_link_list'][i * 3 + 1]
            theta1 = dic['body_link_list'][i * 3 + 2]
            theta2 = res[0]['body_link_list'][i * 3 + 2]
            if len(res) == 2:
                # translational agreement
                x3 = res[1]['body_link_list'][i * 3 + 0]
                y3 = res[1]['body_link_list'][i * 3 + 1]

                # if x1 == [0, 0] and x2 == [0,0] and x3 == [0,0]:
                #     continue
                value = (euclidean_dist_2d(x1[0], x2[0], x1[1], x2[1]) +
                         euclidean_dist_2d(x1[0], x3[0], x1[1], x3[1]) +
                         euclidean_dist_2d(x3[0], x2[0], x3[1], x2[1])) / 3 / head_bone_link_length
                value += (euclidean_dist_2d(y1[0], y2[0], y1[1], y2[1]) +
                          euclidean_dist_2d(y1[0], y3[0], y1[1], y3[1]) +
                          euclidean_dist_2d(y3[0], y2[0], y3[1], y2[1])) / 3 / head_bone_link_length
                body_part_translational_dis[i].append(value)

                # rotational agreement
                theta3 = res[1]['body_link_list'][i * 3 + 2]
                body_part_rotational_dis[i].append((abs(theta1 - theta2) + abs(theta1 - theta3) + abs(theta2 - theta3)) \
                                                   / 3 / 180)
            else:
                if x1 == [0, 0]:
                    continue
                body_part_translational_dis[i].append(
                    euclidean_dist_2d(x1[0], x2[0], x1[1], x2[1]) / head_bone_link_length
                    + euclidean_dist_2d(y1[0], y2[0], y1[1], y2[1]) / head_bone_link_length)

                body_part_rotational_dis[i].append(abs(theta1 - theta2) / 180)

    avg_body_part_translational_dis = [0] * num_body_parts
    avg_body_part_rotational_dis = [0] * num_body_parts

    for i in range(num_body_parts):
        avg_body_part_translational_dis[i] = np.average(body_part_translational_dis[i])
        avg_body_part_rotational_dis[i] = np.average(body_part_rotational_dis[i])

    return avg_body_part_translational_dis, avg_body_part_rotational_dis

def ensemble_uncertainty_quantification():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dataType', type=str, default='val2014_random1k',
                        help='val2014_random1k or val2014_random1k_resolution or val2014_random1k_motion_blur'
                             'or val2014_random1k_dark')
    args = parser.parse_args()


    trans_dis, rotate_dis = compute_transl_rotate_disagreement(args.coco_dataType)


    part_str = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne',
                'Lank', 'Leye', 'Reye', 'Lear', 'Rear', ]
    limb_seq = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
                (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
    limb_seq_str = [f'{part_str[p1]} to {part_str[p2]} ' for p1, p2 in limb_seq]
    evaluation_metrics = [f'transl disagreement {s}' for s in limb_seq_str] \
                         + [f'rotate disagreement {s}' for s in limb_seq_str] \
                         + ['total transl disagreement', 'total rotate disagreement']

    values = trans_dis + rotate_dis + [sum(trans_dis), sum(rotate_dis)]
    df = pd.DataFrame({
        "evaluation_metrics": evaluation_metrics,
        "value": ['%.3f' % v for v in values]
    })
    df.to_csv('eval/results/%s-three-model-epoch100-open-pose-single-scale-1000.csv' % args.coco_dataType)
    print(df)

if __name__ == '__main__':
    # val 2014的所有文件中随机选1000个存到另一个文件夹中去
    # copyFile('dataset/val2014/', 'dataset/val2014_random1k/')
    # df = pd.read_csv('eval/deform_rare_crowded_list.csv')
    # deform_id = set([int(v) for v in df['deformation'].values if not math.isnan(v)])
    # rare_id = set([int(v) for v in df['rare_novel_poses'].values if not math.isnan(v)])
    # crowd_back_id = set([int(v) for v in df['crowded_background'].values if not math.isnan(v)])

    # half_resolution()
    # half_resolution_dataset_annotation()
    motion_blur_dataset()
    dark_dataset()

    # ensemble_uncertainty_quantification()

