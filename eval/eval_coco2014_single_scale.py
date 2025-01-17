import sys

sys.path.append('../dataset/cocoapi/PythonAPI')
sys.path.append("..")
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import code
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import pylab
import os
import os.path
import pandas
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# orderCOCO = [1,0, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4]
orderCOCO = [0, 1, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def load_cmu_val1k(mode):
    # mode == 1: load image id
    # mode == 2: load filename
    flist = []
    f = open('val2014_1k.txt', 'r')
    if mode == 1:
        for line in f:
            flist.append(int(line.split()[mode]))
    else:
        for line in f:
            flist.append(line.split()[mode])
    return flist

def process_single_scale(input_image, model, params, model_params, draw=True):
    oriImg = cv2.imread(input_image)  # B,G,R order

    heatmap_ori_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_ori_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    input_img = np.transpose(np.float32(oriImg[:, :, :, np.newaxis]),
                             (3, 0, 1, 2))  # required shape (1, width, height, channels)

    # image features

    output_blobs = model.predict(input_img)

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    paf = np.squeeze(output_blobs[0])  # output 0 is PAFs

    heatmap_ori_size = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    paf_ori_size = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_ori_size[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_ori_size[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = None
    # body_link_list = []
    if draw:
        canvas = cv2.imread(input_image)  # B,G,R order
        # draw all possible peaks
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

        # draw all detected skeletons
        stickwidth = 4
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    # body_link_list.append([0, 0])
                    # body_link_list.append([0, 0])
                    # body_link_list.append(0)
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                           360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
                # body_link_list.append(X)
                # body_link_list.append(Y)
                # body_link_list.append(angle)

    return canvas, candidate, subset


def compute_keypoints(model_weights_file, cocoGt, output_folder, coco_data_type, epoch_num,
                      fir_img_num, img_id=None, draw=True):
    # load model
    model = get_testing_model()
    model.load_weights(model_weights_file)
    # load model config
    params, model_params = config_reader()

    # load epoch num
    trained_epoch = epoch_num
    # load validation image ids
    if img_id is not None:
        ### TO change
        if fir_img_num != -1:
            imgIds = img_id[:fir_img_num]
        else:
            imgIds = img_id
    elif fir_img_num > 0:
        imgIds = sorted(cocoGt.getImgIds()[:fir_img_num])
    else:
        imgIds = sorted(cocoGt.getImgIds())

    # prepare json output
    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    prediction_folder = '%s/predictions' % (output_folder)
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)

    # prepare json output
    json_fpath = '%s/%s' % (output_folder, outputjson)
    json_file = open(json_fpath, 'w')

    candidate_set = []
    subset_set = []
    image_id_set = []
    counter = 0
    # run keypoints detection per image

    for item in imgIds:
        # load image fname
        if cocoGt is None:
            fname = item
        else:
            fname = cocoGt.imgs[item]['file_name']
        input_fname = '../dataset/%s/%s' % (coco_data_type, fname)
        print(input_fname)
        print('Image file exist? %s' % os.path.isfile(input_fname))

        # run keypoint detection
        visual_result, candidate, subset \
            = process_single_scale(input_fname, model, params, model_params, draw)

        # draw results
        output_fname = '%s/result_%s' % (prediction_folder, fname)
        if draw:
            cv2.imwrite(output_fname, visual_result)
        candidate_set.append(candidate)
        subset_set.append(subset)
        image_id_set.append(item)
        counter = counter + 1

    # dump results to json file
    write_json(candidate_set, subset_set, image_id_set, json_file)


def write_json(candidate_set, subset_set, image_id_set, json_file):
    category_id = 1
    output_data = []
    with json_file as outfile:
        total_images = len(subset_set)
        for i in range(total_images):
            valid_person_num = len(subset_set[i])
            for person in range(valid_person_num):
                valid_parts_num = subset_set[i][person][-1].astype(int)
                keypoints = []
                keypoints_with_neck = []
                score = 0.0
                score_li = []
                for part in range(18):
                    part_idx = orderCOCO[part]
                    if part_idx == 1:
                        # skip neck for coco eval
                        if idx.astype(int) == -1:
                            keypoints_with_neck.append(0)
                            keypoints_with_neck.append(0)
                            keypoints_with_neck.append(0)
                        else:
                            x = candidate_set[i][idx.astype(int), 0].astype(int)
                            y = candidate_set[i][idx.astype(int), 1].astype(int)
                            # score = score + candidate_set[i][idx.astype(int),2]
                            keypoints_with_neck.append(x)
                            keypoints_with_neck.append(y)
                            keypoints_with_neck.append(2)
                    else:
                        idx = subset_set[i][person][part_idx]
                        if idx.astype(int) == -1:
                            keypoints.append(0)
                            keypoints.append(0)
                            keypoints.append(0)
                            score_li.append(0)

                            keypoints_with_neck.append(0)
                            keypoints_with_neck.append(0)
                            keypoints_with_neck.append(0)
                        else:
                            x = candidate_set[i][idx.astype(int), 0].astype(int)
                            y = candidate_set[i][idx.astype(int), 1].astype(int)
                            # score = score + candidate_set[i][idx.astype(int),2]
                            score_li.append(candidate_set[i][idx.astype(int),2])
                            keypoints.append(x)
                            keypoints.append(y)
                            keypoints.append(2)

                            keypoints_with_neck.append(x)
                            keypoints_with_neck.append(y)
                            keypoints_with_neck.append(2)

                # get body link list
                limb_seq = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
                            (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
                body_link_list = []
                # num_keypoints_with_neck = 18
                for l1, l2 in limb_seq:
                    X = [keypoints_with_neck[3*l1], keypoints_with_neck[3*l1+1]]
                    Y = [keypoints_with_neck[3*l2], keypoints_with_neck[3*l2+1]]
                    angle = math.degrees(math.atan2(X[0] - Y[0], X[1] - Y[1]))
                    body_link_list.append(X)
                    body_link_list.append(Y)
                    body_link_list.append(angle)

                # score = score/valid_parts_num.astype(float)
                score = subset_set[i][person][-2]
                json_dict = {"image_id": image_id_set[i][:-4], "category_id": category_id, "keypoints": keypoints,
                             "score": score, "score_list": score_li, "keypoints_with_neck": keypoints_with_neck,
                             "body_link_list": body_link_list}
                output_data.append(json_dict)
        json.dump(output_data, outfile, cls=NpEncoder)


def check_pred_keypoints(pred_keypoint, gt_keypoint, threshold, normalize):
    def get_normalize(input_shape):
        """
        rescale keypoint distance normalize coefficient
        based on input shape, used for PCK evaluation
        NOTE: 6.4 is standard normalize coefficient under
              input shape (256,256)
        # Arguments
            input_shape: input image shape as (height, width)
        # Returns
            scale: normalize coefficient
        """
        # assert input_shape[0] == input_shape[1], 'only support square input shape.'

        # use averaged scale factor for non square input shape
        scale = float((input_shape[0] + input_shape[1]) / 2) / 256.0

        return 6.4 * scale

    # head bone length = abs(gt_Rear - gt_Lear)
    # normalize =

    # check if ground truth keypoint is valid
    if gt_keypoint[0] > 1 and gt_keypoint[1] > 1:
        # calculate normalized euclidean distance between pred and gt keypoints
        distance = np.linalg.norm(gt_keypoint[0:2] - pred_keypoint[0:2]) / normalize
        if distance < threshold:
            # succeed prediction
            return 1
        else:
            # fail prediction
            return 0
    else:
        # invalid gt keypoint
        return -1


def calPCK(pred_path, ann_path, threshold=0.5, invisible_keypoints=False, difficult=False, crowd=False,
           deform=False, deform_id=None, rare=False, rare_id=None, crowd_back=False, crowd_back_id=None):
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    with open(ann_path, 'r') as f:
        ann = json.load(f)['annotations']

    true_cnt = 0
    total_cnt = 0
    for dt in pred:
        left_ear_visibility_index = -4
        right_ear_visibility_index = -1
        if crowd and dt['iscrowd'] == 0:
            continue
        if dt['gtId'] == 0:
            continue
        if deform and dt['image_id'] not in deform_id:
            continue
        if rare and dt['image_id'] not in rare_id:
            continue
        if crowd_back and dt['image_id'] not in crowd_back_id:
            continue

        gt = [a for a in ann if a['id'] == dt['gtId']][0]
        if not gt['keypoints'][left_ear_visibility_index] or not gt['keypoints'][right_ear_visibility_index]:
            continue
        total_cnt += 1

        # left and right ear both labelled, then
        def euclidean_dist_2d(x1, x2, y1, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        head_bone_link_length = euclidean_dist_2d(gt['keypoints'][-2], gt['keypoints'][-5],
                                                  gt['keypoints'][-3], gt['keypoints'][-6])

        dtks = np.array(dt['keypoints']).reshape(-1, 3)
        gtks = np.array(gt['keypoints']).reshape(-1, 3)

        if difficult:
            dtks = np.array(dt['keypoints']).reshape(-1, 3)[8:14]
            gtks = np.array(gt['keypoints']).reshape(-1, 3)[8:14]

        for (dt_x, dt_y, dt_v), (gt_x, gt_y, gt_v) in zip(dtks, gtks):
            if gt_v == 0 or dt_v == 0:
                continue

            if invisible_keypoints and (dt_v != 1 or gt_v != 1):
                continue

            total_cnt += 1
            if euclidean_dist_2d(dt_x, gt_x, dt_y, gt_y) < threshold * head_bone_link_length:
                true_cnt += 1

    print('total cnt', total_cnt)
    return true_cnt / (total_cnt + 1e-5)


def run_eval_metric(cocoGt, prediction_json, coco_dataType, total_time, full_eval, fir_img_num, img_id=None):
    # initialize COCO detections api
    annType = 'keypoints'
    cocoDt = cocoGt.loadRes(prediction_json)
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    # load validation image ids
    if img_id is not None:
        imgIds = img_id
    elif fir_img_num > 0:
        imgIds = sorted(cocoGt.getImgIds()[:fir_img_num])
    else:
        imgIds = sorted(cocoGt.getImgIds())

    out_prefix = 'full'
    if full_eval == False:
        out_prefix = '1k'
        imgIds = load_cmu_val1k(mode=1)

    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    # print('[[evalImgs kankan id is what?', cocoEval.evalImgs)
    cocoEval.accumulate()
    cocoEval.summarize()

    # create output file for accuracy number
    scores = cocoEval.stats
    # serialize to file, to be read
    outputs = np.append(scores, total_time)

    dtIds_all = []
    for dic in cocoEval.evalImgs:
        if dic:
            for a in dic['dtIds']:
                dtIds_all.append(a - 1)
    # dtIds_all = [a - 1 for dic in cocoEval.evalImgs for a in dic['dtIds'] if dic]

    dtIds_match_gtIds = [int(a) for dic in cocoEval.evalImgs if dic for a in
                         dic['dtMatches'][0]]  # 0 means OKS>0.5 then match
    dt_gt_dic = {}
    for dtId, gtId in zip(dtIds_all, dtIds_match_gtIds):
        dt_gt_dic[dtId] = gtId

    ann_path = '../dataset/annotations/person_keypoints_' + coco_dataType + '.json'

    with open(ann_path, 'r') as f:
        data = json.load(f)
    crowded_imgIds = []
    for dic in data['annotations']:
        if dic['iscrowd'] == 1:
            crowded_imgIds.append(dic['image_id'])
    crowded_imgIds = tuple(crowded_imgIds)

    with open(prediction_json) as f:
        pred_data = json.load(f)
        for dtId in range(len(pred_data)):
            pred_data[dtId]['gtId'] = 0 if dtId not in dt_gt_dic else dt_gt_dic[dtId]

            pred_data[dtId]['iscrowd'] = 1 if pred_data[dtId]['image_id'] in crowded_imgIds else 0
        pred_matchGtId_path = prediction_json[:-5] + '_matchGtId.json'
        with open(pred_matchGtId_path, 'w') as final:
            json.dump(pred_data, final, cls=NpEncoder)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    epoch_num = 100
    parser.add_argument('--effect', type=str, default="_motion_blur", help='model weight load, _dark or _motion_blur or _ensemble')
    parser.add_argument('--coco_dataType', '-dt', type=str, default='frames_motion_blur',
                        help='val2014 or val2014_random1k or val2014_random1k_dark or val2014_random1k_motion_blur'
                             'or val2014_random1k_resolution or video_frames_less1mb')
    parser.add_argument('--fir_img_num', type=int, default=200, help='validate on first __ images,'
                                                                    'if -1 means all images')
    parser.add_argument('--compute_keypoint', type=bool, default=True, help='let model predict keypoint or not')
    parser.add_argument('--eval_compute_keypoint', type=bool, default=False,
                        help='evaluate model predicted keypoint or not')

    args = parser.parse_args()
    # load coco eval api
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = 'keypoints'
    prefix = 'person_keypoints'
    print('COCO eval for %s results.' % annType)

    keras_weights_file = '../training/weights%s/weights.%04d.h5' % (args.effect, epoch_num)

    # initialize COCO ground truth api
    cocoGt = None
    if not 'frames' in args.coco_dataType:
        annFile = ('../dataset/annotations/%s_%s.json' % (prefix, args.coco_dataType))
        cocoGt = COCO(annFile)
    tic = time.time()

    # configure img_ids
    img_id = None

    if 'val2014_random1k' in args.coco_dataType:
        img_id = []
        # open file and read the content in a list
        with open('../dataset/val2014_random1k_imgid.txt', 'r') as fp:
            for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
                x = line[:-1]

                # add current item to the list
                img_id.append(int(x))

    if 'frames' in args.coco_dataType:
        img_id = []
        for file_names in os.listdir('../dataset/%s' % args.coco_dataType):
            img_id.append(file_names)

    # eval model
    mode_name = 'open-pose-single-scale'
    output_folder = './results/%s-%s-epoch%d-%s-%s' \
                    % (args.coco_dataType, 'ours' if not args.effect else args.effect[1:], epoch_num,
                       mode_name, args.fir_img_num)
    outputjson = '%s-%s-epoch%d-%s_result.json' \
                    % (args.coco_dataType, 'ours' if not args.effect else args.effect[1:], epoch_num,
                       args.fir_img_num)
    json_path = '%s/%s' % (output_folder, outputjson)

    print('keras_weights_file', keras_weights_file)
    print('output_folder', output_folder)

    # model predict keypoints
    if args.compute_keypoint:
        print('start processing...')

        compute_keypoints(keras_weights_file, cocoGt, output_folder, args.coco_dataType, epoch_num, args.fir_img_num, img_id=img_id,)
        toc = time.time()
        total_time = toc - tic
        print('overall processing time is %.5f' % (toc - tic))

        with open(output_folder + '/total_time.txt', 'w') as f:
            f.write('%s' % total_time)


    # evaluate model predicted keypoints
    if args.eval_compute_keypoint:
        print('Computing keypoints...')
        with open(output_folder + '/total_time.txt', 'r') as f:
            total_time = float(f.readline())

        # run coco eval 2014
        outputs = run_eval_metric(cocoGt, json_path, args.coco_dataType, total_time, full_eval=True,
                                  fir_img_num=args.fir_img_num, img_id=img_id)
        # run coco eval 2014 (1k images random selected by CMU)
        # run_eval_metric(cocoGt, json_path, total_time, full_eval=False)

        # self-annotated robustness metrics
        df = pd.read_csv('deform_rare_crowded_list.csv')
        deform_id = set([int(v) for v in df['deformation'].values if not math.isnan(v)])
        rare_id = set([int(v) for v in df['rare_novel_poses'].values if not math.isnan(v)])
        crowd_back_id = set([int(v) for v in df['crowded_background'].values if not math.isnan(v)])
        # evaluation metrics
        eval_metric_key_names = ['(AP) @[ OKS=0.50:0.95 | area=   all | maxDets= 20 ]',
                                 '(AP) @[ OKS=0.50      | area=   all | maxDets= 20 ]',
                                 '(AP) @[ OKS=0.75      | area=   all | maxDets= 20 ]',
                                 '(AP) @[ OKS=0.50:0.95 | area=medium | maxDets= 20 ]',
                                 '(AP) @[ OKS=0.50:0.95 | area= large | maxDets= 20 ]',
                                 '(AR) @[ OKS=0.50:0.95 | area=   all | maxDets= 20 ]',
                                 '(AR) @[ OKS=0.50      | area=   all | maxDets= 20 ]',
                                 '(AR) @[ OKS=0.75      | area=   all | maxDets= 20 ]',
                                 '(AR) @[ OKS=0.50:0.95 | area=medium | maxDets= 20 ]',
                                 '(AR) @[ OKS=0.50:0.95 | area= large | maxDets= 20 ]',
                                 'prediction time',
                                 'PCK@0.5 all keypoints',
                                 'PCK@0.5 difficult keypoints',
                                 'PCK@0.5 crowded images',
                                 'PCK@0.5 deformation images',
                                 'PCK@0.5 rare pose images',
                                 'PCK@0.5 crowded background images',
                                 ]

        pck_all = calPCK(json_path[:-5] + '_matchGtId.json', annFile)
        pck_difficult = calPCK(json_path[:-5] + '_matchGtId.json', annFile, difficult=True)
        pck_crowd = calPCK(json_path[:-5] + '_matchGtId.json', annFile, crowd=True)
        pck_deform = calPCK(json_path[:-5] + '_matchGtId.json', annFile, deform=True, deform_id=deform_id)
        pck_rare = calPCK(json_path[:-5] + '_matchGtId.json', annFile, rare=True, rare_id=rare_id)
        pck_crowd_back = calPCK(json_path[:-5] + '_matchGtId.json', annFile, crowd_back=True,
                                crowd_back_id=crowd_back_id)

        print('PCK@0.5 all keypoints=', pck_all)
        print('PCK@0.5 difficult keypoints=', pck_difficult)
        print('PCK@0.5 crowded images=', pck_crowd)
        print('PCK@0.5 deformation images=', pck_deform)
        print('PCK@0.5 rare pose images=', pck_rare)
        print('PCK@0.5 crowded background images=', pck_crowd_back)

        values = list(outputs) + [pck_all, pck_difficult, pck_crowd, pck_deform, pck_rare,
                                  pck_crowd_back]
        df = pd.DataFrame({
            "evaluation_metrics": eval_metric_key_names,
            "value": values
        })
        df.to_csv(json_path[:-5] + '.csv')
