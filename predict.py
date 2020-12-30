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

#custom import
# from dataloader.mnist_loader import Mnist_loader
from dataloaders.casting_loader import Castingloader,Castring_transform
from model import f_anogan_dcgan as de
from utils import utils


#set hyper parameter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs = 300
batch_size = 1
lr = 0.0002
save_interval = 5
ndf = 300
ngf = 300
latent_dim = 100
img_size = 300
channels = 3
n_critic = 5
# sample_interval = 400
training_label = 0
split_rate = 0.8
lambda_gp = 10
n_classes = 1
weight_path = 'C:\\Users\LMH\Desktop\personal\\f-AnoGan_mun\\runs'


G = de.Generator(latent_dim=latent_dim,ngf= ngf, channels= channels, bias= True).cuda()
D = de.Discriminator(ndf= ndf, channels = channels, bias=True).cuda()
E = de.Encoder(latent_dim=latent_dim, ndf= ndf, channels= channels, bias= True).cuda()


G.load_state_dict(torch.load('./runs/Generator/G_epoch-242-loss11.046401.pth'))
D.load_state_dict(torch.load('./runs/Discriminator/D_epoch-248-0.000006.pth'))
E.load_state_dict(torch.load('./runs/Autoencoder/_update_min_loss_E_epoch-267.pth'))


composed_transforms_tr = transforms.Compose([
    transforms.ToTensor()])

train_casting = Castingloader(split='def',
                                    transforms=composed_transforms_tr,
                                    )

train_dataloader = DataLoader(train_casting, batch_size=batch_size, shuffle=True)


D.eval()
G.eval()
E.eval()

criterion = nn.MSELoss()
kappa =1
for i,sample in enumerate(train_dataloader):
    # test_imgs = sample['image'].cuda()

    real_imgs = sample['image'].cuda()
    print(real_imgs.shape)
    latent_z = E(real_imgs)
    fake_imgs = G(latent_z)
    fake_feature = D.forward_features(fake_imgs)
    real_feature = D.forward_features(real_imgs)

    real_feature = real_feature / real_feature.max()
    fake_feature = fake_feature / fake_feature.max()

    img_distance = criterion(fake_imgs, real_imgs)
    loss_feature = criterion(fake_feature, real_feature)

    anomaly_score = img_distance + kappa*loss_feature
    utils.compare_images(real_imgs,fake_imgs,i,anomaly_score,reverse=False,threshold=0.3)

    


