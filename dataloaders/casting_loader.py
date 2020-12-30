import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from dataloaders.utils import recursive_glob
# from utils import recursive_glob

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from skimage import io
#custom
from mypath import Path


composed_transforms_tr = transforms.Compose([
    transforms.ToTensor()])

class Castring_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()])
    
    def __call__(self,img):
        img = np.array(img)
        return self.transform

# train_casting = Castingloader(split='train',
#                                     transforms=composed_transforms_tr,
#                                     n_classes = 1)

# train_dataloader = DataLoader(train_casting, batch_size=batch_size, shuffle=True)



class Castingloader(Dataset):
    def __init__(self, root=Path.db_root_dir('casting'),
                split="train",
                transforms = None):
        self.root = root
        self.split = split
        self.transform = transforms
        self.files = {}
        # self.n_classes = n_classes
        self.jpeg_base = os.path.join(self.root,self.split)
        self.annotation_base = os.path.join(self.root,self.split)

        self.files[split] = recursive_glob(rootdir=self.jpeg_base, suffix= '.jpeg')
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.json_base))

        print("Found %d %s files" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        img_path = self.files[self.split][idx].rstrip()
        _img = io.imread(img_path)
        _target = np.ones((1,1))
        if self.transform:
            _img = self.transform(_img)

        sample ={'image':_img, 'label':_target}
        return sample
    
    def generate_batches(self, shuffle=False, drop_last=False, _batch_size = 5):
        dataloader = DataLoader(dataset=self, batch_size=_batch_size,
                                shuffle=shuffle, drop_last=drop_last)
        for sample in dataloader:
            yield sample['image'], sample['label']

if __name__ == '__main__':
    # composed_transforms_tr = transforms.Compose([
    #     transforms.ToTensor()])
    # castingloader = Castingloader(root='C:\\Users\LMH\Desktop\personal\\f-AnoGan_mun\dataset\casting_data',split='train',transforms=composed_transforms_tr,)
    for ii, sample in enumerate(castingloader):
        img = sample['image'].numpy()
        img = np.squeeze(img)
        print(img.shape)
        print('\nlabel=========================\n')
        print(sample['label'].shape)
        break