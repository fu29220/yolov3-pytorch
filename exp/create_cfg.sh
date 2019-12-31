#!/bin/sh

bash ../../config/create_custom_model.sh 1 yolov3-cocoPerson.cfg

echo "person" > classes.names

echo "
classes= 1
train=exp/cocoPerson/train/train.json
valid=exp/cocoPerson/val/val.json
names=exp/cocoPerson/classes.names
" > cocoPerson.data

ln -sf /data/fushilian/dataset/cocoPerson/train
ln -sf /data/fushilian/dataset/cocoPerson/val

