# coding=utf-8
"""
@File: temp.py
@Time: 2019/12/23
@Author: Zengrui Zhao
""" 
import numpy as np
import matplotlib.pyplot as plt
img = np.load('/home/zzr/Project/xjw/data/train/Labels/train_1.npy')
plt.imshow(img[..., 1], cmap='jet')
plt.show()
# print(np.unique(img[..., 0]), np.unique(img[..., 0]))