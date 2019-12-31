from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def weighted_binary_cross_entropy(output, target, weights=[100, 1]):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))

def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3

def create_modules(module_defs):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())

        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
            modules.add_module('MaxPool2d', maxpool)

        elif mdef['type'] == 'upsample':
            #modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')
            upsample = Upsample(scale_factor=int(mdef["stride"]), mode="nearest")
            modules.add_module(f"upsample_{i}", upsample)

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[j + 1 if j > 0 else j] for j in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            #layers = [int(x) for x in mdef["layers"].split(",")]
            #filters = sum([output_filters[1:][i] for i in layers])
            #modules.add_module(f"route_{i}", EmptyLayer())
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])
            #filters = output_filters[1:][int(mdef["from"])]
            #modules.add_module(f"shortcut_{i}", EmptyLayer())

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            anchor_idxs = [int(x) for x in mdef["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in mdef["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(mdef["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{i}", yolo_layer)

            #yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            #modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
            #                    num_classes=int(mdef['classes']),  # number of classes
            #                    img_dim=img_size)  # (416, 416)
            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            arc = 'default'
            #try:
            if arc == 'defaultpw' or arc == 'Fdefaultpw':  # default with positive weights
                b = [-4, -3.6]  # obj, cls
            elif arc == 'default':  # default no pw (40 cls, 80 obj)
                b = [-5.5, -5.0]
            elif arc == 'uBCE':  # unified BCE (80 classes)
                b = [0, -9.0]
            elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                b = [10, -0.1]
            elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                b = [-2.1, -1.8]
            elif arc == 'uFBCE' or arc == 'uFBCEpw':  # unified FocalBCE (5120 obj, 80 classes)
                b = [0, -6.5]
            elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                b = [7.7, -1.1]

            bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
            bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
            bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
            # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
            module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
                # utils.print_model_biases(model)
            #except:
            #    print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list, routs


#def create_modules(module_defs):
#    """
#    Constructs module list of layer blocks from module configuration in module_defs
#    """
#    hyperparams = module_defs.pop(0)
#    output_filters = [int(hyperparams["channels"])]
#    module_list = nn.ModuleList()
#    for module_i, module_def in enumerate(module_defs):
#        modules = nn.Sequential()
#
#        if module_def["type"] == "convolutional":
#            bn = int(module_def["batch_normalize"])
#            filters = int(module_def["filters"])
#            kernel_size = int(module_def["size"])
#            pad = (kernel_size - 1) // 2
#            modules.add_module(
#                f"Conv2d",
#                nn.Conv2d(
#                    in_channels=output_filters[-1],
#                    out_channels=filters,
#                    kernel_size=kernel_size,
#                    stride=int(module_def["stride"]),
#                    padding=pad,
#                    bias=not bn,
#                ),
#            )
#            if bn:
#                modules.add_module(f"BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
#            if module_def["activation"] == "leaky":
#                modules.add_module(f"activation", nn.LeakyReLU(0.1))
#            elif module_def['activation'] == 'swish':
#                modules.add_module('activation', Swish())
#
#        elif module_def["type"] == "maxpool":
#            kernel_size = int(module_def["size"])
#            stride = int(module_def["stride"])
#            if kernel_size == 2 and stride == 1:
#                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
#            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
#            modules.add_module(f"maxpool_{module_i}", maxpool)
#
#        elif module_def["type"] == "upsample":
#            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
#            modules.add_module(f"upsample_{module_i}", upsample)
#
#        elif module_def["type"] == "route":
#            layers = [int(x) for x in module_def["layers"].split(",")]
#            filters = sum([output_filters[1:][i] for i in layers])
#            modules.add_module(f"route_{module_i}", EmptyLayer())
#
#        elif module_def["type"] == "shortcut":
#            filters = output_filters[1:][int(module_def["from"])]
#            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
#
#        elif module_def["type"] == "yolo":
#            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
#            # Extract anchors
#            anchors = [int(x) for x in module_def["anchors"].split(",")]
#            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
#            anchors = [anchors[i] for i in anchor_idxs]
#            num_classes = int(module_def["classes"])
#            img_size = int(hyperparams["height"])
#            # Define detection layer
#            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
#            modules.add_module(f"yolo_{module_i}", yolo_layer)
#        # Register module list and number of output filters
#        module_list.append(modules)
#        output_filters.append(filters)
#
#    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, dtype=torch.float, device='cuda').repeat(g, 1).view([1, 1, g, g])
        self.grid_y = torch.arange(g, dtype=torch.float, device='cuda').repeat(g, 1).t().view([1, 1, g, g])
        #self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.scaled_anchors = FloatTensor(self.anchors) / self.stride
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            #.contiguous()
        )


        # Get outputs
        #  x = torch.sigmoid(prediction[..., 0])  # Center x
        #  y = torch.sigmoid(prediction[..., 1])  # Center y
        #  w = prediction[..., 2]  # Width
        #  h = prediction[..., 3]  # Height
        #  pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        #  pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        sig_idx = list(range(prediction.size(-1)))
        sig_idx.pop(2)
        sig_idx.pop(2)
        prediction[...,sig_idx] = torch.sigmoid(prediction[...,sig_idx])

        # If grid size does not match current we compute new offsets
        #  if grid_size != self.grid_size:
            #  self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        #  pred_boxes = FloatTensor(prediction[..., :4].shape)
        #  #  pred_boxes[..., 0] = x.data + self.grid_x
        #  #  pred_boxes[..., 1] = y.data + self.grid_y
        #  #  pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        #  #  pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        #  pred_boxes[..., 0] = x.data
        #  pred_boxes[..., 1] = y.data
        #  pred_boxes[..., 2] = w.data
        #  pred_boxes[..., 3] = h.data
    
        """
        output = torch.cat(
            (
                #  pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_boxes.view(num_samples, -1, 4),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        """
        #  output = [pred_boxes, pred_conf, pred_cls]
        
        ### 
        #  self.shape_box=pred_boxes.size()
        #  self.shape_cls=pred_cls.size()
        #  self.shape_conf=pred_conf.size()

        #  if targets is None:
            #  return output, 0
        #  else:
            #  iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                #  pred_boxes=pred_boxes,
                #  pred_cls=pred_cls,
                #  target=targets,
                #  anchors=self.scaled_anchors,
                #  ignore_thres=self.ignore_thres,
            #  )

            #  # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            #  loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            #  loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            #  loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            #  loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            #  loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            #  loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            #  loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            #  loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            #  total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            #  # Metrics
            #  cls_acc = 100 * class_mask[obj_mask].mean()
            #  conf_obj = pred_conf[obj_mask].mean()
            #  conf_noobj = pred_conf[noobj_mask].mean()
            #  conf50 = (pred_conf > 0.5).float()
            #  iou50 = (iou_scores > 0.5).float()
            #  iou75 = (iou_scores > 0.75).float()
            #  detected_mask = conf50 * class_mask * tconf
            #  precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            #  recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            #  recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            #  self.metrics = {
                #  "loss": to_cpu(total_loss).item(),
                #  "x": to_cpu(loss_x).item(),
                #  "y": to_cpu(loss_y).item(),
                #  "w": to_cpu(loss_w).item(),
                #  "h": to_cpu(loss_h).item(),
                #  "conf": to_cpu(loss_conf).item(),
                #  "cls": to_cpu(loss_cls).item(),
                #  "cls_acc": to_cpu(cls_acc).item(),
                #  "recall50": to_cpu(recall50).item(),
                #  "recall75": to_cpu(recall75).item(),
                #  "precision": to_cpu(precision).item(),
                #  "conf_obj": to_cpu(conf_obj).item(),
                #  "conf_noobj": to_cpu(conf_noobj).item(),
                #  "grid_size": grid_size,
            #  }

            #  return output, total_loss
        return prediction


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list, self.routs = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if len(layer)>0 and hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = weighted_binary_cross_entropy
        self.cross_loss = nn.CrossEntropyLoss()
        self.obj_scale = 1
        self.noobj_scale = 100

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layers = [int(j) for j in module_def['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[j] for j in layers], 1)
                    except:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[j] for j in layers], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = x + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                #  x, layer_loss = module[0](x, targets, img_dim)
                x = module[0](x, targets, img_dim)
                #  loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x if i in self.routs else [])
        #  yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        #yolo_outputs = torch.cat(yolo_outputs, 1)

        ###
        #  if loss == 0:
            #  loss_ = loss
        #  else:
            #  loss_ = loss.type(torch.cuda.FloatTensor) # multi gpu train need cuda tensor
        #  yolo_outputs_gpu = yolo_outputs.cuda()

        #  return yolo_outputs_gpu if targets is None else (loss, yolo_outputs_gpu)
        #print("**** yolo_outputs[0,0,0,0,0,:10]", yolo_outputs[2][0,0,0,0,:10])
        return yolo_outputs


    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
