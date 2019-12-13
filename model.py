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


class SEResNext50(nn.Module):
    def __init__(self, n_classes=6):
        super(SEResNext50, self).__init__()
        src_net = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000,
                                                                  pretrained='imagenet')
        modules = list(src_net.children())[:-2]
        modules.append(nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=1),
                                     nn.Conv2d(2048, 4096, kernel_size=1, stride=1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(4096, n_classes, kernel_size=1, stride=1)))
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        out = self.features(x)

        return out


if __name__ == '__main__':
    net = SEResNext50()
    print(net)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = net.to(device)
    # summary(net, (3, 144, 144))
    # aa = torch.randn((1, 3, 144, 144)).to(device)
    # print(net(aa).size())