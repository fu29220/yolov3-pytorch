import os
import sys
import random

import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pycocotools.coco as coco

from utils.augmentations import horisontal_flip


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class COCODataset(Dataset):
    def __init__(self, json_path, img_size=416, augment=True, multiscale=True):
        super(COCODataset, self).__init__()
        self.img_dir = os.path.join(json_path.rsplit('/', 1)[0], 'images')
        self.coco = coco.COCO(json_path)
        self.images = self.coco.getImgIds()
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)


        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        h, w = img.shape[1:]
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        padded_h, padded_w = img.shape[1:]

        # ---------
        #  Label
        # ---------
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        targets = None
        #  if os.path.exists(label_path):
        if len(anns) > 0:
            #  boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes = torch.tensor([[ann['category_id']-1]+ann['bbox'] for ann in anns])
            # Extract coordinates for unpadded + unscaled image
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]/2 + pad[0])/padded_w
            boxes[:, 2] = (boxes[:, 2]+ boxes[:, 4]/2 + pad[2])/padded_h
            boxes[:, 3] /= padded_w
            boxes[:, 4] /= padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        r = self.img_size / padded_w
        pad_scale = list(pad) + [r] 
        return img_id, img, targets, pad_scale

    def collate_fn(self, batch):
        paths, imgs, targets, pad_scales = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets, pad_scales

    def __len__(self):
        return len(self.coco.imgs)
