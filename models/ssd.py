import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from layers import *
#from data import voc, coco
import os


class SSD(nn.Module):


    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        pass

    def forward(self, x):

        pass

    def load_weights(self, base_file):
        pass


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v=='M':
            layers +=[nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v=='C':
            layers +=[nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v , kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers +=[conv2d, nn.ReLU(inplace=True)]
            in_channels= v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
    conv6 = nn.Conv2d(512,1024,kernel_size=3, padding=6,dilation=6)
    conv7 = nn.Conv2d(1024,1024,kernel_size=1)
    layers +=[pool5,conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

    pass


def add_extras(cfg, i, batch_norm=False):
    pass


def multibox(vgg, extra_layers, cfg, num_classes):
    pass


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    pass