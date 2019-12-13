"""
test how many workers should be used and whether using pin_memory
"""
import torch
import sys
from torchvision import datasets, transforms
import time
from torch.utils.data import DataLoader
sys.path.append('..')
from src.data import Data

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    for num_workers in range(4, 32, 4):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_set = Data(rootpth='/home/zzr/Data/Skin', mode='train')
        train_loader = DataLoader(train_set,
                                  batch_size=128,
                                  drop_last=True,
                                  shuffle=True,
                                  **kwargs)
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST('./data', train=True, download=True,
        #                    transform=transforms.Compose([
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.1307,), (0.3081,))
        #                    ])),
        #     batch_size=64, shuffle=True, **kwargs)

        start = time.time()
        for epoch in range(1, 2):
            for batch_idx, (data, target) in enumerate(train_loader):  # 不断load
                pass

        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
