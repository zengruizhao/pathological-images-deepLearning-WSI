# coding=utf-8
"""
@File    : inference.py
@Time    : 2019/12/11
@Author  : Zengrui Zhao
"""
import tqdm
import time
import os
from src.readWSI import WSIPyramid
import numpy as np
import torchvision.transforms as transforms
from src.model import SEResNext50
import matplotlib.pyplot as plt
import torch
import argparse
from PIL import Image
import torch.nn.functional as F

using_level = 1  # 20x
img_size = 5264
trainImageSize = 144
downsampling = 32
stride = img_size - trainImageSize + downsampling
# stride = int(img_size / 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataPth', type=str, default='/home/zzr/Data/wsi')
    parse.add_argument('--model', type=str,
                       default='/home/zzr/Project/skinAreaSegmentation/model_out/191212-192559_SEResNext50/out_10.pth')
    parse.add_argument('--savePth', type=str, default='./')

    return parse.parse_args()


def transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                    (0.12062634, 0.1206659, 0.12071837))])


def inference(wsi, model):
    """
    test img
    :param wsi:
    :param model:
    :return:
    """
    for bounding_box in wsi.bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2] + 1)
        b_y_end = int(bounding_box[1]) + int(bounding_box[3] + 1)
        mag_factor = 2 ** (wsi.level - using_level)
        col_cords = np.arange(int(b_x_start * mag_factor / stride),
                              int(np.ceil(b_x_end * mag_factor / stride)))
        row_cords = np.arange(int(b_y_start * mag_factor / stride),
                              int(np.ceil(b_y_end * mag_factor / stride)))
        #
        model.eval()
        with torch.no_grad():
            tempOutput = model(torch.rand((1, 3, img_size, img_size)).to(device))
            tempOutputShape = tempOutput.shape[-1]
            print('OPTs output shape:{}'.format(tempOutputShape))
            result = np.ones((len(row_cords) * tempOutputShape,
                              len(col_cords) * tempOutputShape)) * 4
            row, column = len(row_cords) * tempOutputShape, len(col_cords) * tempOutputShape
            for idx, i in tqdm.tqdm(enumerate(col_cords)):
                for jdx, j in enumerate(row_cords):
                    patchSizeX = min(img_size, wsi.slide.level_dimensions[using_level][0] - stride * i)
                    patcyhSizeY = min(img_size, wsi.slide.level_dimensions[using_level][1] - stride * j)
                    img = np.array(wsi.slide.read_region((stride * i * 2 ** using_level,
                                                          stride * j * 2 ** using_level),
                                                         using_level,
                                                         (patchSizeX, patcyhSizeY)))[..., 0:-1]
                    img = Image.fromarray(img)
                    img = transform()(img).unsqueeze(0).to(device)
                    output = F.softmax(model(img), dim=1).cpu().detach().numpy().squeeze()
                    # output = model(img).cpu().detach().numpy().squeeze()
                    # output = output[0, ...].squeeze()
                    output = np.argmax(output, axis=0)
                    if output.shape[0] != tempOutputShape:
                        row = jdx * tempOutputShape + output.shape[0]
                    if output.shape[1] != tempOutputShape:
                        column = idx * tempOutputShape + output.shape[1]
                    result[jdx*tempOutputShape: jdx*tempOutputShape + output.shape[0],
                           idx*tempOutputShape: idx*tempOutputShape + output.shape[1]] = output

        result = result[0:row, 0:column]
        return result


def main(args):
    # device = torch.device('cpu')
    model = SEResNext50().to(device)
    model.load_state_dict(torch.load(args.model))
    WSI = [i for i in os.listdir(args.dataPth) if i.endswith('ndpi')]
    WSI = ['2018-06-06 15.12.38.ndpi']
    for i in WSI:
        wsi = WSIPyramid(os.path.join(args.dataPth, i))
        result = inference(wsi, model)
        # np.save(os.path.join(args.savePth, 'result.npy'), result)
        plt.imshow(result, cmap=plt.get_cmap('jet'))
        plt.show()
        # print(result)


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    main(args)
    print('Done! time cost: {:.4f}'.format(time.time() - start))