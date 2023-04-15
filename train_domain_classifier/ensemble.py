import json
import numpy as np
import os
import shutil
import argparse

def ensembled_prediction(data_type):
    dark_path = '../eval/results/%s-dark-epoch100-open-pose-single-scale-1000/' \
                '%s-dark-epoch100-1000_result.json' \
                % (data_type, data_type)
    mb_path = '../eval/results/%s-motion_blur-epoch100-open-pose-single-scale-1000/' \
              '%s-motion_blur-epoch100-1000_result.json' \
              % (data_type, data_type)
    ours_path = '../eval/results/%s-ours-epoch100-open-pose-single-scale-1000/' \
                '%s-ours-epoch100-1000_result.json' \
                % (data_type, data_type)
    with open(ours_path, 'r') as f:
        ours_json = json.load(f)
    with open(mb_path, 'r') as f:
        motion_blur_json = json.load(f)
    with open(dark_path, 'r') as f:
        dark_json = json.load(f)

    cls_path = './results/%s.json' % data_type
    with open(cls_path, 'r') as f:
        cls_json = json.load(f)

    output_folder = '../eval/results/%s-ensemble-epoch100-open-pose-single-scale-1000' % data_type
    ours_folder = '../eval/results/%s-ours-epoch100-open-pose-single-scale-1000/' % data_type
    mb_folder = '../eval/results/%s-motion_blur-epoch100-open-pose-single-scale-1000/' % data_type
    dark_folder = '../eval/results/%s-dark-epoch100-open-pose-single-scale-1000/' % data_type
    prediction_folder = '%s/predictions/' % (output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)

    ensembled_res = []
    for dic in cls_json:
        img_id = dic['image_id']
        classifications = np.argmax(dic['effect_prob'])
        if classifications == 0:
            res = list(filter(lambda d: d['image_id'] == img_id, ours_json))

            file_name = 'result_COCO_val2014_%012d.jpg' % img_id
            shutil.copyfile(ours_folder + 'predictions/' + file_name, prediction_folder + file_name)

        elif classifications == 1:
            res = list(filter(lambda d: d['image_id'] == img_id, dark_json))

            file_name = 'result_COCO_val2014_%012d.jpg' % img_id
            shutil.copyfile(dark_folder + 'predictions/' + file_name, prediction_folder + file_name)
        else:
            res = list(filter(lambda d: d['image_id'] == img_id, motion_blur_json))

            file_name = 'result_COCO_val2014_%012d.jpg' % img_id
            shutil.copyfile(mb_folder + 'predictions/' + file_name, prediction_folder + file_name)

        # copy prediction file from
        ensembled_res += res

    ensembled_res_path = output_folder + '/%s-ensemble-epoch100-1000_result.json' % data_type
    with open(ensembled_res_path, 'w') as f:
        json.dump(ensembled_res, f)

    total_time = 0.0
    with open(ours_folder + 'total_time.txt', 'r') as f:
        total_time += float(f.readline())
    with open(dark_folder + 'total_time.txt', 'r') as f:
        total_time += float(f.readline())
    with open(mb_folder + 'total_time.txt', 'r') as f:
        total_time += float(f.readline())

    with open(output_folder + '/total_time.txt', 'w') as f:
        f.write('%s' % total_time)

def ensembled_prediction_matchGtId(data_type):
    output_folder = '../eval/results/%s-ensemble-epoch100-open-pose-single-scale-1000' % data_type

    dark_path = '../eval/results/%s-dark-epoch100-open-pose-single-scale-1000/' \
                '%s-dark-epoch100-1000_result_matchGtId.json' \
                % (data_type, data_type)
    mb_path = '../eval/results/%s-motion_blur-epoch100-open-pose-single-scale-1000/' \
              '%s-motion_blur-epoch100-1000_result_matchGtId.json' \
              % (data_type, data_type)
    ours_path = '../eval/results/%s-ours-epoch100-open-pose-single-scale-1000/' \
                '%s-ours-epoch100-1000_result_matchGtId.json' \
                % (data_type, data_type)
    with open(ours_path, 'r') as f:
        ours_json = json.load(f)
    with open(mb_path, 'r') as f:
        motion_blur_json = json.load(f)
    with open(dark_path, 'r') as f:
        dark_json = json.load(f)

    cls_path = './results/%s.json' % data_type
    with open(cls_path, 'r') as f:
        cls_json = json.load(f)

    ensembled_res = []
    for dic in cls_json:
        img_id = dic['image_id']
        classifications = np.argmax(dic['effect_prob'])
        if classifications == 0:
            res = list(filter(lambda d: d['image_id'] == img_id, ours_json))
        elif classifications == 1:
            res = list(filter(lambda d: d['image_id'] == img_id, dark_json))

        else:
            res = list(filter(lambda d: d['image_id'] == img_id, motion_blur_json))

        # copy prediction file from
        ensembled_res += res

    ensembled_res_path = output_folder + '/%s-ensemble-epoch100-1000_result_matchGtId.json' % data_type

    with open(ensembled_res_path, 'w') as f:
        json.dump(ensembled_res, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dataType', type=str, default='val2014_random1k_resolution',
                        help='val2014_random1k or val2014_random1k_resolution or val2014_random1k_motion_blur'
                             'or val2014_random1k_dark')
    args = parser.parse_args()

    ensembled_prediction(args.coco_dataType)
    ensembled_prediction_matchGtId(args.coco_dataType)
