# run evaluation on coco2014 dataset, use single scale approach for fast development
# accumulate ensembled predictions
cd ../train_expert_classifier
datatype='val2014_random1k_resolution'
python ensemble.py --coco_dataType $datatype

cd ../eval
python eval_coco2014_multi_modes.py --effect _ensemble --coco_dataType $datatype --fir_img_num 1000 --compute_keypoint False --eval_compute_keypoint True