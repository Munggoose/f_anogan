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

#custom import
from dataloader.mnist_loader import Mnist_loader
from model import f_anogan_dcgan as de

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

writer = SummaryWriter(logdir='runs/Gan_training')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_epochs = 200
batch_size = 64
lr = 0.0002
ndf = 28
ngf = 28
latent_dim = 100
img_size = 28
channels = 1
n_critic = 5
sample_interval = 400
training_label = 0
split_rate = 0.8
lambda_gp = 10


train_dataset = MNIST('./', train=True, download=True)

_x_train = train_dataset.data[train_dataset.targets == 1]
x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)), dim=0)

_y_train = train_dataset.targets[train_dataset.targets == 1]
y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)), dim=0)

train_mnist = Mnist_loader(x_train, y_train,
                                transform=transforms.Compose(
                                    [transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
                                )

train_dataloader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)


g_net = de.Generator(latent_dim=latent_dim,ngf= ngf, channels= channels, bias= True).cuda()
d_net = de.Discriminator(ndf= ndf, channels = channels, bias=True).cuda()

optimizer_G = optim.Adam(g_net.parameters(), lr = lr, weight_decay=1e-5)
optimizer_D = optim.Adam(d_net.parameters(), lr = lr, weight_decay=1e-5)

criterion = nn.BCELoss()

padding_epoch = len(str(n_epochs)) #3
padding_i = len(str(len(train_dataloader))) # 2

for epoch in range(n_epochs):
    for i,(imgs,_) in enumerate(train_dataloader):
        real_imgs = imgs.cuda()

        #train discriminator

        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1,1).cuda()

        fake_img = g_net(z).cuda()

        real_validity = d_net(real_imgs)
        fake_validity = d_net(fake_img)

        #adversarial loss
        d_loss_real = criterion(real_validity, torch.ones_like(real_validity).cuda())
        d_loss_fake = criterion(fake_validity, torch.zeros_like(fake_validity).cuda())
        d_loss = d_loss_real +d_loss_fake

        d_loss.backward()

        optimizer_D.step()
        if i % n_critic == 0 :
            #train generator
            optimizer_G.zero_grad()
            fake_imgs = g_net(z)
            fake_validity = d_net(fake_imgs)
            
            g_loss = criterion(fake_validity,torch.ones_like(fake_validity).cuda())
            g_loss.backward()
            optimizer_G.step()

            # d_losses.append(d_loss)
            # g_losses.append(g_lass)
            
            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                f"[D loss: {d_loss.item():3f}] "
                f"[G loss: {g_loss.item():3f}]")