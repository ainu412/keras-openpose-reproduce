# keras-openpose-reproduce

This is a keras implementation of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) and [DOPE-Uncertainty](https://github.com/NVlabs/DOPE-Uncertainty).


## Prerequisites

  0. Keras and Tensorflow (tested on Linux machine)
  0. Python3
  0. GPU with at least `11GB` memory
  0. More than `250GB` of disk space for training data

Please also install the following packages:

    $ pip install libboost-all-dev libhdf5-serial-dev libzmq3-dev libopencv-dev python-opencv python3-tk python-imaging
    $ pip install Cython scikit-image pandas zmq h5py opencv-python IPython configobj


## Download COCO 2014 Dataset

Please download the COCO dataset and the official COCO evaluation API. Go to folder `dataset` and simply run the following commands:

    $ cd dataset
    $ ./step1_download_coco2014.sh
    $ ./step2_setup_coco_api.sh


## Prepare Training Data 

Before model training, we convert the images to the specific data format for efficient training. We generate the heatmaps, part affinity maps, and then convert them to HDF5 files. Go to the folder `training`, and run the scripts. The process takes around 2 hours.

    $ cd training
    $ python3 generate_masks_coco2014.py
    $ python3 generate_hdf5_coco2014.py

After this, you will generate `train_dataset_2014.h5` and `val_dataset_2014.h5`. The files are about `182GB` and `3.8GB`, respectively.

## Training of Human Pose Keypoints Detector

train Human Pose Keypoints Detector in three domains seperately

    $ cd training
    # train original model
    $ python train_pose.py --effect ""
    # train dark domain model
    $ python train_pose.py --effect _dark
    # train motion blur domain model
    $ python train_pose.py --effect _motion_blur

You can find our trained models at [Dropbox](https://www.dropbox.com/sh/k0yh5efafzzgvy9/AAB3OWBxq38JBIPXe3wBz3KXa?dl=0)

## Training, Validation, and Prediction of Domain Classifier

Train Domain Classifier to identify between three kinds of images: normal image, dark image (image with low lightning), and motion blur image (image with high degree of motion blur)

```
$ cd train_domain_classifier
# train model and save to 'train_domain_classifier/log/2023-04-08_12-40/checkpoints/model_{epoch+1:04d}.pth'
$ python train.py
$ python validate.py
```

validation set loss: 0.556 | validation set accuracy: 99.5%

You can find our trained model at [Dropbox](https://www.dropbox.com/sh/247969lxme1dzrp/AABh5kW3AXL2mcU_8LZh4U8Aa?dl=0)



Predict and save the predicted domain to `train_domain_classifier/results/val2014_random1k_resolution.json` file

```
# predict image domain with our trained model on val2014_random1k_resolution dataset
$ cd train_domain_classifier
$ python ensemble.py --coco_dataType val2014_random1k_resolution
```

parameter:

 `--coco_dataType`: dataset domain

Range: `"val2014_random1k", "val2014_random1k_dark", "val2014_random1k_motion_blur", "val2014_random1k_resolution"`

## Evaluation on COCO Keypoints Datasets

Only implement openpose single scale for fast development

    # predict keypoints and evaluate predicted results origianl model on first 100 images of dark validation dataset
    $ cd eval
    $ python eval_coco2014_single_scale.py --effect "" --coco_dataType val2014_random1k_dark --fir_img_num 100
    
    # predict keypoints and evaluate predicted results dark model on first 1000 images of low resolution validation dataset
    $ python eval_coco2014_single_scale.py --effect _dark --coco_dataType val2014_random1k_resolution --fir_img_num 1000

parameters:

1. `--effect`: specifies neural network type

​		Range: `"", "_dark", "_motion_blur", "_ensemble"`

2.  `--coco_dataType`: dataset domain

   Range: `"val2014_random1k", "val2014_random1k_dark", "val2014_random1k_motion_blur", "val2014_random1k_resolution"`

3. `--fir_img_num`: , evaluate on first __ number of images

   Range: `∈[1, 1000]`

4. `--compute keypoint`

​		Range: `True, False`

5. `--eval_compute_keypoint`

​		Range: `True, False`

## Evaluate on three models' uncertainty quantification

Compute three models' uncertainty quantification -- translational and rotational disagreement, and save the results to `eval/results/%s-three-model-epoch100-open-pose-single-scale-1000.csv' % args.coco_dataType` file

```
$ python util_ainu.py --coco_dataType val2014_random1k_resolution
```

parameter:

 `--coco_dataType`: dataset domain

Range: `"val2014_random1k", "val2014_random1k_dark", "val2014_random1k_motion_blur", "val2014_random1k_resolution"`


## Evaluation of Ensemble of Networks on COCO Keypoints Datasets

```
# compute ensemble prediction results on val2014_random1k_resolution COCO keypoints dataset
$ cd train_expert_classifier
$ python ensemble.py --coco_dataType val2014_random1k_resolution

# evaluate ensemble results on val2014_random1k_resolution COCO keypoints dataset
$ cd ../eval
$ python eval_coco2014_single_scale.py --effect _ensemble --coco_dataType val2014_random1k_resolution --fir_img_num 1000 --compute_keypoint False --eval_compute_keypoint True
```

You may find prediction results at [Dropbox](https://www.dropbox.com/sh/mt52o9rqi5ggyn1/AAC2oxGVPbVimvMF-zmkdM6Ea?dl=0)

## Acknowledgment

This repo is based upon  [kevinlin311tw](https://github.com/kevinlin311tw)'s repo [keras-openpose-reproduce](https://github.com/kevinlin311tw/keras-openpose-reproduce), [@anatolix](https://github.com/anatolix)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation), and [@michalfaber](https://github.com/michalfaber)'s repo [keras_Realtime_Multi-Person_Pose_Estimation](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)


## Citation

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
    
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }





Difficult detection cases by original dataset annotations: 
difficult joints: knees & ankles & hips [Rhip, Rkne, Rank, Lhip, Lkne, Lank]
crowded: multi-person in proximityDifficult detection cases with complex images by my own annotations (according to visual perception and human intelligence):
deformation: weird, deformed and not elegant pose with negative sentiment
rare and novel poses: interesting or funny pose with positive sentiment. (Poses that are rare in dataset pose distribution and it might be hard for a person to stay in this pose for long.)
complex/crowded background: not clear background, not pronounced entity, not blurred background, many (in terms of quantity and categories) objectsDifficult detection cases with low quality image
low resolution: half the width and half the height of original images
motion blur: add horizontal motion blur to images
low lightning: lower image brightness
