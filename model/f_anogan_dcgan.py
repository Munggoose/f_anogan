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


class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf = 224, channels=1, bias = True):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*16, kernel_size=4, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(ngf*16),
            nn.LeakyReLU(0.2, inplace=True),
            #4
            nn.ConvTranspose2d(ngf*16, ngf*16, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*16),
            nn.LeakyReLU(0.2, inplace=True),
            #8
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            #16
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            #32
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            #64
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            #128
            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.Tanh()
            #256
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=28, channels=3, bias=True): 
        super().__init__()
        
        def discriminator_block(in_features, out_features, bn=True):
            if bn:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.BatchNorm2d(out_features),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            else:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 3, bias=bias),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.features = nn.Sequential(
            *discriminator_block(channels, ndf, bn=False),
            *discriminator_block(ndf, ndf*2, bn=True),
            *discriminator_block(ndf*2, ndf*4, bn=True),
            *discriminator_block(ndf*4, ndf*8, bn=True),
            *discriminator_block(ndf*8, ndf*16, bn=True),
            *discriminator_block(ndf*16, ndf*16, bn=True)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward_features(self, x):
        features = self.features(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        validity = self.last_layer(features)
        return validity


#  latent_dim=100, ndf = 28, channels=1,
class Encoder(nn.Module):
    def __init__(self, latent_dim=100, ndf=28, channels=3, bias=True):
        super().__init__()
        
        def encoder_block(in_features, out_features, bn=True):
            if bn:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 1, bias=bias),
                         nn.BatchNorm2d(out_features),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            else:
                block = [
                         nn.Conv2d(in_features, out_features, 4, 2, 3, bias=bias),
                         nn.LeakyReLU(0.2, inplace=True)
                ]
            return block

        self.features = nn.Sequential(
            *encoder_block(channels, ndf, bn=False),
            *encoder_block(ndf, ndf*2, bn=True),
            *encoder_block(ndf*2, ndf*4, bn=True),
            *encoder_block(ndf*4, ndf*8, bn=True),
            *encoder_block(ndf*8, ndf*16, bn=True),
            *encoder_block(ndf*16, ndf*16, bn=True),
            nn.Conv2d(ndf*16, latent_dim, 4, 1, 0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        validity = self.features(x)
        return validity


if __name__ == "__main__":
    latent_dim =100
    ngf= 224
    bias =True
    channels = 3 
    z = torch.randn(1, latent_dim, 1,1)
    batch =nn.BatchNorm2d(ngf*4)
    relu =nn.LeakyReLU(0.2, inplace=True)    
    G = Generator(latent_dim=100, ngf = 224, channels=3, bias = True)
    result = G(z)
    print(result.shape)
