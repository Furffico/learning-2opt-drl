import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


#######################################
# Dataset
#######################################
class TSPDataset(Dataset):

    def __init__(self, dataset_fname=None, size=50, num_samples=10, seed=None, device='cpu'):
        """
        json: {
            Points: { [id:string]: number[][] }
            OptTour: { [id:string]: number[] }
            OptDistance: { [id:string]: number }
        }
        """
        super(TSPDataset, self).__init__()

        self.data_set = []      # 点的坐标
        self.opt = []           # 最短路径的距离
        if seed is not None:
            random.seed(seed)
        if dataset_fname is not None:
            print('  [*] Loading dataset from {}'.format(dataset_fname))
            dset = pd.read_json(dataset_fname)
            # 从数据集中随机采样num_samples个样本（点的个数可以不同）
            ids = random.sample(range(len(dset)), num_samples)
            for i in tqdm(ids):
                self.data_set.append(torch.tensor(dset.iloc[i, 0],device = device))
                self.opt.append(dset.iloc[i, -1])
        else:
            # randomly sample points uniformly from [0, 1]^2
            for i in range(num_samples):
                x = torch.FloatTensor(size, 2).uniform_(0, 1)
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
