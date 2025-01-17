import sys
# sys.path.append('../dataset/cocoapi/PythonAPI')
sys.path.append('/data/dataset/cocoapi/PythonAPI')
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import argparse

def preproc(mode, effect):
    
    dataset_dir = '../dataset'#os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset'))

    val_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_%s2014.json" % mode)

    val_images_dir = os.path.join(dataset_dir, "%s2014%s" % (mode, effect))
    val_masks_dir = os.path.join(dataset_dir, "%smask2014%s" % (mode, effect))

    if not os.path.exists(val_masks_dir):
        os.mkdir(val_masks_dir)

    coco = COCO(val_anno_path)
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)

        img_path = os.path.join(val_images_dir, "COCO_%s2014_%012d.jpg" %(mode,img_id))
        mask_miss_path = os.path.join(val_masks_dir, "mask_miss_%012d.png" % img_id)
        mask_all_path = os.path.join(val_masks_dir, "mask_all_%012d.png" % img_id)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        mask_all = np.zeros((h, w), dtype=np.uint8)
        mask_miss = np.zeros((h, w), dtype=np.uint8)
        flag = 0
        for p in img_anns:
            seg = p["segmentation"]

            if p["iscrowd"] == 1:
                mask_crowd = coco.annToMask(p)
                temp = np.bitwise_and(mask_all, mask_crowd)
                mask_crowd = mask_crowd - temp
                flag += 1
                continue
            else:
                mask = coco.annToMask(p)

            mask_all = np.bitwise_or(mask, mask_all)

            if p["num_keypoints"] <= 0:
                mask_miss = np.bitwise_or(mask, mask_miss)

        if flag<1:
            mask_miss = np.logical_not(mask_miss)
        elif flag == 1:
            mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
            mask_all = np.bitwise_or(mask_all, mask_crowd)
        else:
            raise Exception("crowd segments > 1")

        cv2.imwrite(mask_miss_path, mask_miss * 255)
        cv2.imwrite(mask_all_path, mask_all * 255)

        if (i % 1000 == 0):
            print("Processed %d of %d" % (i, len(ids)))

    print("Done !!!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--effect', type=str, default='', help='_dark or _motion_blur')
    args = parser.parse_args()

    preproc('train', args.effect)
    preproc('val', args.effect)
