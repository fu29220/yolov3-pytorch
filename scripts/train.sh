#!/bin/bash

# gpu ids
DEVICE_IDS=0,1,2
export CUDA_VISIBLE_DEVICES=$DEVICE_IDS

YOLO_PKG_DIR=$(dirname $(
    cd "$(dirname "$0")"
    pwd
))

EXP_DIR=${YOLO_PKG_DIR}/exp/cocoPerson

export PYTHONPATH=${YOLO_PKG_PATH}:$PYTHONPATH
# python -W ignore::UserWarning train.py \
python ${YOLO_PKG_DIR}/yolov3/train.py \
    --model-def ${YOLO_PKG_DIR}/configs/yolov3-spp-1cls.cfg \
    --data-cfg ${YOLO_PKG_DIR}/exp/cocoPerson/cocoPerson.data \
    --weights ${YOLO_PKG_DIR}/weights/ultralytics68.pt \
    --device-ids ${DEVICE_IDS} \
    --exp-dir ${EXP_DIR} \
    --epochs 100 \
    --batch-size 96 \
    --eval-interval 1 \
    --ckpt-interval 20
