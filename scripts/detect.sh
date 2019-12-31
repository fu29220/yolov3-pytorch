#!/bin/sh

# gpu ids
DEVICE_IDS=0,1,2,3
export CUDA_VISIBLE_DEVICES=$DEVICE_IDS

python -W ignore::UserWarning detect.py \
    --model_def exp/cocoPerson/yolov3-cocoPerson.cfg \
    --weights_path /home/fushilian/PyTorch-YOLOv3_bk2/checkpoints/yolov3_ckpt_90.pth \
    --class_path exp/cocoPerson/classes.names \
    --image_folder exp/cocoPerson/imgs \
    --nms_thres 0.3 \
    --conf_thres 0.9
