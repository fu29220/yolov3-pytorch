#!/bin/sh

# gpu ids
DEVICE_IDS=0,1,2
export CUDA_VISIBLE_DEVICES=$DEVICE_IDS

python -W ignore::UserWarning test.py \
    --model_def ../yolov3/cfg/yolov3-spp-1cls.cfg \
    --data_config exp/cocoPerson/cocoPerson.data \
    --weights_path ../yolov3/weights/backup30.pt \
    --batch_size 72 \
    --device_ids ${DEVICE_IDS} \
    --conf_thres 0.2

    #--iou_thres 0.5 \
    #--conf_thres 0.8 \
    #--nms_thres 0.4 \
