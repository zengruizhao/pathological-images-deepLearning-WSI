# -*- coding: utf-8 -*-
"""
@File    : data.py
@Time    : 2019/12/10
@Author  : Zengrui Zhao
"""
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os.path as osp
import os
import cv2
from tqdm import tqdm
import numpy as np
import pickle
from PIL import Image
import PIL
import matplotlib.pyplot as plt


class Data(Dataset):
    def __init__(self, rootpth='/home/zzr/Data/Skin',
                 des_size=(144, 144),
                 mode='train'):
        """
        :param rootpth: 根目录
        :param re_size: 数据同一resize到这个尺寸再后处理
        :param mode: train/val/test
        """
        self.root_path = rootpth
        self.des_size = des_size
        self.mode = mode
        self.name = None
        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        # 处理对应标签
        assert (mode == 'train' or mode == 'val' or mode == 'test')

        # 读取文件名称
        self.file_names = []
        for root, dirs, names in os.walk(osp.join(rootpth, mode)):
            for name in names:
                self.file_names.append(osp.join(root, name))
        random.shuffle(self.file_names)
        # 确定分隔符号
        self.split_char = '\\' if '\\' in self.file_names[0] else '/'

        # totensor 转换n
        if mode == 'train':
            self.to_tensorImg = \
                transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomApply([transforms.RandomRotation(90)], p=.3),
                                    transforms.RandomApply([transforms.ColorJitter(.1, .1, .1)], p=.3),
                                    transforms.FiveCrop(128),
                                    transforms.Lambda(lambda crops: torch.stack(
                                        [transforms.ToTensor()(crop) for crop in crops])),
                                    transforms.Lambda(lambda crops: torch.stack(
                                        [transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                              (0.12062634, 0.1206659, 0.12071837))(crop)
                                         for crop in crops]))

            ])
        else:
            self.to_tensorImg = transforms.Compose([transforms.Resize(128),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                                         (0.12062634, 0.1206659, 0.12071837))])

    def __getitem__(self, idx):
        self.name = self.file_names[idx]
        category = int(self.name.split(self.split_char)[-2])
        img = Image.open(self.name)
        # a = np.array(img)
        # plt.imshow(img)
        # plt.show()
        return self.to_tensorImg(img), torch.tensor([category]*5) if self.mode == 'train' else torch.tensor(category)

    def __len__(self):
        return len(self.file_names)

    def get_mean_std(self, type='train', mean_std_path='../mean.pkl'):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        num_imgs = len(self.file_names)
        for data in tqdm(self.file_names):
            img = np.array(Image.open(data)) / 255.
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))

        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stdevs, f)
            print('pickle done')


if __name__ == '__main__':
    data = Data(mode='train')
    # for i in tqdm(range(len(data))):
    #     a, b = data.__getitem__(i)
    #     print(a.shape, b.shape)
    #     for j in range(5):
    #         img = a[j, ...]
    #         img = np.transpose(np.array(img), (1, 2, 0))
    #         print(b[j])
    #         plt.imshow(img)
    #         plt.show()
    #     break

    data.get_mean_std()