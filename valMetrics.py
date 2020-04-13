# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from data import Data
from model import SEResNext50, Vgg, Vgg_padding
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size', type=int, default=256)
    parse.add_argument('--num_workers', type=int, default=12)

    parse.add_argument('--data_dir', type=str, default='/media/tiger/Data/zzr/skin')
    parse.add_argument('--predefinedModel', type=str, default='./model_out/200316-174019_vgg_padding/out_49.pth')# 200312-101152_VGG 48
    return parse.parse_args()

def main_worker(args):
    val_set = Data(rootpth=args.data_dir, mode='test')
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=args.num_workers)

    net = Vgg_padding().to(device)
    net.load_state_dict(torch.load(args.predefinedModel))
    net.eval()
    correct, total = 0, 0
    tp, tp_fp, tp_fn = [0] * 6, [0] * 6, [0] * 6
    with torch.no_grad():
        for img, lb in tqdm(val_loader):
            img, lb = img.to(device), np.array(lb)
            outputs = net(img).view(img.shape[0], -1)
            outputs = F.softmax(outputs, dim=1)
            predicted = np.array(torch.max(outputs, dim=1)[1].cpu())
            for i in range(6):
                array = np.where(predicted==i)
                tp[i] += (lb[array] == i).sum()
                tp_fp[i] += (predicted == i).sum()
                tp_fn[i] += (lb == i).sum()
            correct += (predicted == lb).sum()
            total += len(lb)

    sen, spe = [0] * 6, [0] * 6
    acc = correct * 1. / total
    eps = 0.001
    for i in range(6):
        sen[i] = (tp[i] + eps) / (tp_fp[i] + eps)
        spe[i] = (tp[i] + eps) / (tp_fn[i] + eps)

    print(acc, spe, sen)

if __name__ == '__main__':
    args = parse_args()
    main_worker(args)
