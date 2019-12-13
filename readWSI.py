# coding=utf-8
"""
read WSI images and extract small patches using multi-process
@File    : readWSI.py
@Time    : 2019/12/11
@Author  : Zengrui Zhao
"""
import cv2
import numpy as np
import os
from multiprocessing import Process
from skimage.filters import threshold_otsu, try_all_threshold, threshold_mean
from skimage.morphology import reconstruction
from skimage.measure import regionprops, find_contours, label
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import skimage
from skimage import io
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

img_WSI_dir = '/home/zzr/Data/wsi'
fore_path = '/home/zzr/Data/wsi'
# Parameters
nProcs = 1  # the number of process
stride = 144  # set stride 36
using_level = 1  # 0: max level; 1
patch_size = (144, 144)     # 144


def hole_fill(img):
    """
    like the function of imfill in Matlab
    :param img:
    :return:
    """
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    mask = img
    return reconstruction(seed, mask, method='erosion').astype('uint8')


def max_prop(img):
    """
    select the max area
    :param img:
    :return:
    """
    label_, label_num = label(np.uint8(img), return_num=True)
    props = regionprops(label_)
    filled_area = []
    label_list = []
    for prop in props:
        filled_area.append(prop.area)
        label_list.append(prop.label)
    filled_area_sort = np.sort(filled_area)
    true_label = label_list[np.squeeze(np.argwhere(filled_area == filled_area_sort[-1]))]
    img = (label_ == true_label).astype('uint8')

    return img


class WSIPyramid:
    def __init__(self, path, stride=1):
        self.stride = stride
        self.slide, downsample_image, self.level, m, n = self.read_image(path)
        self.bounding_boxes, rgb_contour, self.image_dilation = \
            self.find_roi_bbox_1(downsample_image, show=False)

    def get_bbox(self, cont_img, rgb_image=None, show=False):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_contour = None
        if rgb_image is not None:
            rgb_contour = rgb_image.copy()
            line_color = (255, 0, 0)  # blue color code
            cv2.drawContours(rgb_contour, contours, -1, line_color, 1)
            if show:
                plt.imshow(rgb_contour)
                plt.show()

        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return bounding_boxes, rgb_contour

    def find_roi_bbox(self, rgb_image):   # bgr
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        thres = threshold_mean(hsv[..., 0])
        # fig, ax = try_all_threshold(hsv[..., 0])
        # plt.show()
        mask = (hsv[..., 0] > thres).astype('uint8')

        close_kernel = np.ones((5, 5), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((7, 7), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        image_fill = hole_fill(image_open)
        image_fill = max_prop(image_fill)
        bounding_boxes, rgb_contour = self.get_bbox(np.array(image_fill), rgb_image=rgb_image, show=False)
        return bounding_boxes, rgb_contour, image_fill

    def find_roi_bbox_1(self, rgb_image, show=False):     # rgb
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        thres = threshold_mean(hsv[..., 0])
        # fig, ax = try_all_threshold(hsv[..., 0])
        # plt.show()
        mask = (hsv[..., 0] > thres).astype('uint8')

        # close_kernel = np.ones((11, 11), dtype=np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        # open_kernel = np.ones((7, 7), dtype=np.uint8)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        image_open = cv2.morphologyEx(np.array(image_open),
                                      cv2.MORPH_DILATE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        image_fill = hole_fill(image_open)
        image_fill = max_prop(image_fill)
        bounding_boxes, rgb_contour = self.get_bbox(np.array(image_fill), rgb_image, show)

        return bounding_boxes, rgb_contour, image_fill

    def read_image(self, image_path):
        try:
            image = OpenSlide(image_path)
            w, h = image.dimensions
            n = int(math.floor((h - 0) / self.stride))
            m = int(math.floor((w - 0) / self.stride))
            level = image.level_count - 1
            if level > 7:
                level = 7
            downsample_image = np.array(image.read_region((0, 0), level, image.level_dimensions[level]))[..., 0:-1]
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None, None

        return image, downsample_image, level, m, n


def get_patch(startCol, endCol, row_cords, stride, image_dilation,
              mag_factor, slide, save_path):
    col_cords = np.arange(startCol, endCol)
    for i in col_cords:
        print(i, '/', col_cords[-1])
        for j in row_cords:
            if ((stride * i) <= (slide.level_dimensions[using_level][0] - patch_size[0])) \
                    & ((stride * j) <= (slide.level_dimensions[using_level][1] - patch_size[0])) \
                    & int(image_dilation[min(stride * j // mag_factor, image_dilation.shape[0] - 1),
                                      min(stride * i // mag_factor, image_dilation.shape[1] - 1)]) != 0:

                    img = skimage.img_as_float(
                        np.array(slide.read_region((stride * i * 2 ** using_level, stride * j * 2 ** using_level),
                                                   using_level, patch_size))).astype(np.float32)[..., 0:-1]

                    name = str(stride * i) + '_' + str(stride * j) + '.jpg'
                    # print np.mean(img)
                    io.imsave(os.path.join(save_path, name), img)


def main():
    start = time.time()
    ProcessPointer = [None] * nProcs
    WSI = [i for i in os.listdir(img_WSI_dir) if i.endswith('ndpi')]
    # WSI = ['566827 - 2018-07-30 15.00.33.ndpi']
    for name1 in WSI:
        wsi = WSIPyramid(os.path.join(img_WSI_dir, name1), stride)
        out_path = ''.join(name1.split('.')[0:-1])
        save_path = os.path.join(fore_path, out_path)  # if np.mean(img) < 0.9 else back_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # print('{} rows, {} columns'.format(n, m))
        print('%s Classification is in progress' % name1)
        for bounding_box in wsi.bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2]+1)
            b_y_end = int(bounding_box[1]) + int(bounding_box[3]+1)
            mag_factor = 2 ** (wsi.level - using_level)  # level-1: 20x

            col_cords = np.arange(int(b_x_start*mag_factor/stride), int(b_x_end*mag_factor/stride))
            row_cords = np.arange(int(b_y_start*mag_factor/stride), int(b_y_end*mag_factor/stride))
            ColPerCore = int(math.ceil(len(col_cords) / nProcs))
            for proc in range(nProcs):
                startCol = col_cords[0] + ColPerCore * proc
                endCol = min(col_cords[0] + ColPerCore * (proc + 1), len(col_cords))
                ProcessPointer[proc] = Process(target=get_patch, args=(
                        startCol, endCol, row_cords, stride, wsi.image_dilation,
                        mag_factor, wsi.slide, save_path))
                ProcessPointer[proc].start()

            for proc in range(nProcs):
                ProcessPointer[proc].join()

    print('Time:', time.time() - start)


if __name__ == '__main__':
    main()
