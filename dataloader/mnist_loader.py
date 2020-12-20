import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import random


class Mnist_loader(Dataset):
    def __init__(self, data, labels, transform=None):


        self.transform = transform
        self.data = data
        self.labels = labels
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels


# class Mnist_loader(Dataset):
#     def __init__(self, split_rate,transform=None):

#         train_dataset = MNIST('./', train=True, download=True)

#         _x_train = train_dataset.data[train_dataset.targets == 1]
#         x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)), dim=0)

#         _y_train = train_dataset.targets[train_dataset.targets == 1]
#         y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)), dim=0)

#         self.transform = transform
#         self.train_data = x_train
#         self.train_labels = y_train
#         self.test_data = x_test_normal
#         self.test_labels = y_test_normal
    

#     def __len__(self):
#         return len(self.train_data),len(self.test_data)
        
#     def __getitem__(self, idx):
#         tr ={ 'img': self.train_data[idx], 'labels':self.train_labels[idx]}
#         ts ={'img': self.test_data[idx],'labels':self.test_labels[idx]}
        
#         if self.transform:
#             tr['img'] = self.transform(tr['img'])
#             ts['img'] = self.transform(ts['img'])

#         return tr,ts