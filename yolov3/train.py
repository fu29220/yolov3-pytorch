from __future__ import division

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import *
#  from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate



def main(opt):

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    output_dir = opt.exp_dir+"/output"
    ckpt_dir = opt.exp_dir+"/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_cfg)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    
    # If specified we start from checkpoint
    if opt.weights:
        if opt.weights.endswith(".pt"):
            print("******* load pretrained weights", opt.weights)
            chkpt = torch.load(opt.weights, map_location=device)
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
            #model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.weights)
    else:
        model.apply(weights_init_normal)
        
    device_ids = range(len(opt.device_ids.split(',')))
    if is_cuda:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        #model.yolo_layers = model.module.yolo_layers

    # Get dataloader
    dataset = COCODataset(train_path, augment=True, multiscale=opt.multi_scale)
    print("******len(dataset)", len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    pg0, pg1 = [], []
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]
        else:
            pg0 += [v]
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = optim.SGD(pg0, lr=0.000125, momentum=0.95, nesterov=False)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.000484})
    del pg0, pg1


    loss_ma=0.
    for epoch in range(1, opt.epochs+1):
        model.train()
        start_time = time.time()
        nb = len(dataloader)
        pbar = tqdm.tqdm(enumerate(dataloader), total=nb)
        for batch_i, (_, imgs, targets, _) in pbar:
            batch_i += 1

            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = model(imgs, targets)
            loss = compute_loss(model.module if is_cuda else model, targets, outputs, imgs.size(2))
            loss.mean().backward()

            batches_done = len(dataloader) * (epoch-1) + batch_i
            if batches_done % opt.grad_accum:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            loss_ma = ((batches_done-1)*loss_ma+loss.mean().item())/batches_done
            log_str = "[Epoch %d/%d, Batch %d/%d, Loss %.4f]" % (epoch, opt.epochs, batch_i, len(dataloader), loss_ma)
            pbar.set_description(log_str)

            if is_cuda:
                model.module.seen += imgs.size(0)
            else:
                model.seen += imgs.size(0)


        if epoch % opt.eval_interval == 0:
            print("\n---- Evaluating Model ----")
            evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.01,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )


        if epoch % opt.ckpt_interval == 0:
            torch.save(model.module.state_dict(), "{}/yolov3_ckpt_{%d}.pth".format(ckpt_dir, epoch))


def get_args():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='number of epochs')

    parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help='image batch size')

    parser.add_argument(
            '--grad-accum',
            type=int,
            default=2,
            help='number of gradient accums before step')

    parser.add_argument(
            '--model-def',
            type=str,
            default='configs/yolov3.cfg',
            help='path to model definition file')
    
    parser.add_argument(
            '--data-cfg',
            type=str,
            default='configs/coco.data',
            help='path to data config file')
    
    parser.add_argument(
            '--weights',
            type=str,
            default='',
            help='if specified starts from checkpoint model')
   
    parser.add_argument(
            '--workers',
            type=int,
            default=8,
            help='number of cpu threads to use during batch generation')
    
    parser.add_argument(
            '--img-size',
            type=int,
            default=416,
            help='size of each image dimension')
    
    parser.add_argument(
            '--ckpt-interval',
            type=int,
            default=1,
            help='interval between saving model weights')

    parser.add_argument(
            '--eval-interval',
            type=int,
            default=1,
            help='interval evaluations on validation set')
    
    parser.add_argument(
            '--compute-map',
            type=bool,
            default=False,
            help='if True computes mAP every tenth batch')

    parser.add_argument(
            '--multi-scale',
            type=bool,
            default=True,
            help='allow for multi-scale training')

    parser.add_argument(
            '--device-ids',
            type=str,
            default='0',
            help='gpu device ids')

    parser.add_argument(
            '--exp-dir',
            type=str,
            default='.',
            help='experiment dir of input-and-output data')
    
    # yapf: enable
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    #  logger = Logger("logs")
    main(args)
    
