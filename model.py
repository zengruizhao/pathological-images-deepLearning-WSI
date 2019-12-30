# -*- coding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/10
@Author  : Zengrui Zhao
"""

import torch
import torch.nn as nn
import pretrainedmodels
from torchsummary import summary
from torchvision.models import vgg19_bn
cfgs = {'1': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']}


class SEResNext50(nn.Module):
    def __init__(self, classes=6):
        super(SEResNext50, self).__init__()
        src_net = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000,
                                                                  pretrained='imagenet')
        modules = list(src_net.children())[:-2]
        modules.append(nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=1)))
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Conv2d(2048, classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)

        return out


class Vgg(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.features = self.make_layers()
        self.classifier = nn.Conv2d(1024, classes, kernel_size=1)

    def make_layers(self):
        layers = []
        in_channels = 3
        for v in cfgs['1']:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.Conv2d(in_channels, 512, kernel_size=3),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True),
                   nn.AvgPool2d(kernel_size=2, stride=1)]
        layers += [nn.Conv2d(512, 1024, kernel_size=1),
                   nn.BatchNorm2d(1024),
                   nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)

        return out


if __name__ == '__main__':
    net = Vgg()
    print(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    summary(net, (3, 128, 128))
    # aa = torch.randn((2, 3, 128, 128)).to(device)
    # print(net(aa).size())