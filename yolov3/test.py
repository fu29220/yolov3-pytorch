from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
from progress.bar import Bar

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from pycocotools.cocoeval import COCOeval


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = COCODataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    results=[]

    bar = Bar("coco eval", max=len(dataloader))
    for batch_i, (img_ids, imgs, _, pad_scales) in enumerate(dataloader):

        num_samples = imgs.size(0)
        img_dim = imgs.size(2)
       
        with torch.no_grad():
            outputs = model(imgs.cuda())


            ### 
            y=[]
            num_classes = outputs[0].size(-1)-5
            for i, output in enumerate(outputs):
                yolo_layer = model.module.yolo_layers[i]
            
                pred_boxes = output[..., :4]
                pred_conf = output[..., 4]
                pred_cls = output[..., 5:]
                if num_classes==1:
                    pred_cls[...,-1]=1.

                stride = img_dim/pred_boxes.size(2)
                _, anchor_w, anchor_h, grid_x, grid_y = compute_grid_offsets(img_dim, pred_boxes.size(2), yolo_layer.anchors)
                
                pred_boxes[..., 0] += grid_x
                pred_boxes[..., 1] += grid_y
                pred_boxes[..., 2] = torch.exp(pred_boxes[..., 2]) * anchor_w
                pred_boxes[..., 3] = torch.exp(pred_boxes[..., 3]) * anchor_h
                
                output = torch.cat([pred_boxes.view(num_samples, -1, 4) * stride, pred_conf.view(num_samples, -1, 1),pred_cls.view(num_samples, -1, num_classes)],-1)
                y.append(output)
            y = torch.cat(y, 1) 


            y = non_max_suppression(y, conf_thres=conf_thres, nms_thres=nms_thres)
            
            for j,bbs in enumerate(y):
                if isinstance(bbs,torch.Tensor):
                    #bbs = rescale_boxes(bbs, img_dim, pad_scales[j][5:])
                    bbs = bbs.cpu().numpy()
                else:
                    bbs = []
                for bb in bbs:
                    bb[:4] /= pad_scales[j][-1]
                    bb[[0,2]] -= pad_scales[j][0]
                    bb[[1,3]] -= pad_scales[j][2]
                    bb[[2,3]] -= bb[[0,1]]
                    
                    res={'image_id':img_ids[j], 'category_id': 1, 'bbox': bb[:4], 'score': bb[4]}
                    results.append(res)
        bar.next()

    bar.finish()

    coco_dets = dataset.coco.loadRes(results)
    coco_eval = COCOeval(dataset.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--device_ids", default="0", help="gpu device ids")
    opt = parser.parse_args()
    print(opt)

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    valid_dir = data_config["valid_dir"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    #print(type(model.state_dict()), model.state_dict().keys())
    if opt.weights.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights)
    else:
        # Load checkpoint weights
        chkpt = torch.load(opt.weights, map_location=device)
        if 'model' in chkpt:
            chkpt = chkpt['model']
        #chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        chkpt = {k: v for k, v in chkpt.items() if model.state_dict()[k].numel() == v.numel()}
        #model.load_state_dict(chkpt['model'], strict=False)
        model.load_state_dict(chkpt)

    device_ids = range(len(opt.device_ids.split(',')))
    if is_cuda:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    print("Compute mAP...")

    evaluate(
        model,
        path=valid_path,
        img_dir=valid_dir,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )
