# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import time
import datetime
import argparse
import os
import os.path as osp
from data import Data
from logger import get_logger
from model import SEResNext50


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=20)
    parse.add_argument('--schedule_step', type=int, default=5)

    parse.add_argument('--batch_size', type=int, default=192//5)
    parse.add_argument('--test_batch_size', type=int, default=256)
    parse.add_argument('--num_workers', type=int, default=12)

    parse.add_argument('--eval_fre', type=int, default=1)
    parse.add_argument('--msg_fre', type=int, default=10)
    parse.add_argument('--save_fre', type=int, default=1)

    parse.add_argument('--name', type=str, default='SEResNext50')
    parse.add_argument('--data_dir', type=str, default='/home/zzr/Data/Skin')
    parse.add_argument('--log_dir', type=str, default='../logs')
    parse.add_argument('--tensorboard_dir', type=str, default='../tensorboard')
    parse.add_argument('--model_out_dir', type=str, default='../model_out')
    parse.add_argument('--model_out_name', type=str, default='final_model.pth')
    parse.add_argument('--seed', type=int, default=5, help='random seed')
    parse.add_argument('--predefinedModel', type=str, default='./')
    return parse.parse_args()


def evalute(net, val_loader, epoch, logger):
    logger.info('------------after epo {}, eval...-----------'.format(epoch))
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.cuda(), lb.cuda()
            outputs = net(img).view(img.shape[0], -1)
            outputs = F.softmax(outputs, dim=1)
            predicted = torch.max(outputs, dim=1)[1]
            correct += (predicted == lb).sum().cpu().item()
            total += lb.size()[0]

    acc = correct * 1. / total
    logger.info('accuracy:{:.4f} / epoch{}'.format(acc, epoch))
    # writer.add_scalar('acc', acc, epoch)
    net.train()

    return acc


def main_worker(args, logger):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        writer = SummaryWriter(logdir=args.sub_tensorboard_dir)
        train_set = Data(rootpth=args.data_dir, mode='train')
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=False,
                                  num_workers=args.num_workers)

        val_set = Data(rootpth=args.data_dir, mode='test')
        val_loader = DataLoader(val_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=args.num_workers)

        net = SEResNext50().to(device)
        # x = torch.autograd.Variable(torch.rand(1, 3, 144, 144)).to(device)
        # writer.add_graph(net, x)
        # net.load_state_dict(torch.load(args.predefinedModel))
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
        #                                                  factor=0.5,
        #                                                  patience=3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.schedule_step, gamma=.1)
        loss_record = []
        iter = 0
        running_loss = []
        st = glob_st = time.time()
        total_iter = len(train_loader) * args.epoch
        acc = 0
        for epoch in range(args.epoch):
            if epoch != 0 and epoch % args.eval_fre == 0:
                acc = evalute(net, val_loader, epoch, logger)
                writer.add_scalar('acc', acc, iter)

            if epoch != 0 and epoch % args.save_fre == 0:
                model_out_name = osp.join(args.sub_model_out_dir, 'out_{}.pth'.format(epoch))
                # 防止分布式训练保存失败
                state_dict = net.modules.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state_dict, model_out_name)

            for img, lb in train_loader:
                iter += 1
                bs, ncrops, c, h, w = img.size()
                img = img.to(device).view(-1, c, h, w)
                lb = lb.to(device).view(-1)
                # for i in range(img.shape[0]):
                #     print(lb[i])
                #     imgshow = np.transpose(img[i, ...], (1, 2, 0))
                #     plt.imshow(imgshow)
                #     plt.show()

                # img, lb = img.cuda(), lb.cuda()
                optimizer.zero_grad()
                outputs = net(img).view(img.shape[0], -1)
                loss = criterion(outputs, lb)
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

                if iter % args.msg_fre == 0:
                    # writer.add_images('Image', img, iter)
                    # for name, param in net.named_parameters():
                    #     writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)

                    ed = time.time()
                    spend = ed - st
                    global_spend = ed - glob_st
                    st = ed

                    eta = int((total_iter - iter) * (global_spend / iter))
                    eta = str(datetime.timedelta(seconds=eta))
                    global_spend = str(datetime.timedelta(seconds=(int(global_spend))))

                    avg_loss = np.mean(running_loss)
                    loss_record.append(avg_loss)
                    running_loss = []

                    lr = optimizer.param_groups[0]['lr']

                    msg = '. '.join([
                        'epoch:{epoch}',
                        'iter/total_iter:{iter}/{total_iter}',
                        'lr:{lr:.5f}',
                        'loss:{loss:.4f}',
                        'spend/global_spend:{spend:.4f}/{global_spend}',
                        'eta:{eta}'
                    ]).format(
                        epoch=epoch,
                        iter=iter,
                        total_iter=total_iter,
                        lr=lr,
                        loss=avg_loss,
                        spend=spend,
                        global_spend=global_spend,
                        eta=eta
                    )
                    logger.info(msg)
                    writer.add_scalar('loss', avg_loss, iter)
                    writer.add_scalar('lr', lr, iter)

            scheduler.step()
        # 训练完最后评估一次
        evalute(net, val_loader, args.epoch, logger)

        out_name = osp.join(args.sub_model_out_dir, args.model_out_name)
        torch.save(net.cpu().state_dict(), out_name)

        logger.info('-----------Done!!!----------')

    except:
        logger.exception('Exception logged')
    finally:
        writer.close()


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 唯一标识
    unique_name = time.strftime('%y%m%d-%H%M%S_') + args.name
    args.unique_name = unique_name

    # 每次创建作业使用不同的tensorboard目录
    args.sub_tensorboard_dir = osp.join(args.tensorboard_dir, args.unique_name)
    # 保存模型的目录
    args.sub_model_out_dir = osp.join(args.model_out_dir, args.unique_name)

    # 创建所有用到的目录
    for sub_dir in [args.sub_tensorboard_dir,
                    args.sub_model_out_dir,
                    args.log_dir]:
        if not osp.exists(sub_dir):
            os.makedirs(sub_dir)

    log_file_name = osp.join(args.log_dir, args.unique_name + '.log')
    logger = get_logger(log_file_name)

    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main_worker(args, logger=logger)
