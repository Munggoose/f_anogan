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
import os

#custom import
# from dataloader.mnist_loader import Mnist_loader
from dataloaders.casting_loader import Castingloader
from model import f_anogan_dcgan as de
from utils import utils

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False


#set hyper parameter
writer = SummaryWriter(logdir='runs/Gan_training')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs = 1
batch_size = 5
lr = 0.0002
save_interval = 5
ndf = 28
ngf = 28
latent_dim = 500
img_size = 256
channels = 3
n_critic = 5
# sample_interval = 400
training_label = 0
split_rate = 0.8
lambda_gp = 10
n_classes = 1
weight_path = 'C:\\Users\LMH\Desktop\personal\\f-AnoGan_mun\\runs'


#load dataloader
# train_dataset = MNIST('./', train=True, download=True)

# _x_train = train_dataset.data[train_dataset.targets == 1]
# x_train, x_test_normal = _x_train.split((int(len(_x_train) * split_rate)), dim=0)

# _y_train = train_dataset.targets[train_dataset.targets == 1]
# y_train, y_test_normal = _y_train.split((int(len(_y_train) * split_rate)), dim=0)

# train_mnist = Mnist_loader(x_train, y_train,
#                                 transform=transforms.Compose(
#                                     [transforms.ToPILImage(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.5], [0.5])])
#                                 )

composed_transforms_tr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()])



# train_casting = Castingloader(split='train',
#                                     transforms=composed_transforms_tr,
#                                     )

# train_dataloader = DataLoader(train_casting, batch_size=batch_size, shuffle=True)

train_casting = Castingloader(split='train',
                                    transforms=composed_transforms_tr,
                                    )

train_dataloader = DataLoader(train_casting, batch_size=batch_size, shuffle=True)


#setting net
G = de.Generator(latent_dim=latent_dim,ngf= ngf, channels= channels, bias= True).cuda()
D = de.Discriminator(ndf= ndf, channels = channels, bias=True).cuda()
E = de.Encoder(latent_dim=latent_dim, ndf= ndf, channels= channels, bias= True).cuda()



optimizer_G = optim.Adam(G.parameters(), lr = lr, weight_decay=1e-5)
optimizer_D = optim.Adam(D.parameters(), lr = lr, weight_decay=1e-5)

criterion = nn.BCELoss()

padding_epoch = len(str(n_epochs)) #3
padding_i = len(str(len(train_dataloader))) # 2

min_loss = 9999999
min_d_loss = 999999
min_g_loss = 999999

# train Generator & Dicirminator
G.train()
D.train()
E.train()

for epoch in range(n_epochs):
    for i,sample in enumerate(train_dataloader):
        real_imgs = sample['image'].cuda()


        #train discriminator

        optimizer_D.zero_grad()
        # z = torch.randn(batch_size, latent_dim, 1,1).cuda()
        z = torch.randn(batch_size, latent_dim, 1,1).cuda()


        fake_img = G(z).cuda()


        real_validity = D(real_imgs)
        fake_validity = D(fake_img)

        #adversarial loss
        d_loss_real = criterion(real_validity, torch.ones_like(real_validity).cuda())
        d_loss_fake = criterion(fake_validity, torch.zeros_like(fake_validity).cuda())
        d_loss = d_loss_real + d_loss_fake
        if d_loss < min_d_loss:
            min_d_loss = d_loss
            d_save_path = os.path.join(weight_path,'Discriminator',f'D_epoch-{epoch}-{d_loss:4f}.pth')
            torch.save(D.state_dict(),d_save_path)
            

        d_loss.backward()
        optimizer_D.step()
        if i % n_critic == 0 :
            #train generator
            optimizer_G.zero_grad()
            fake_imgs = G(z)
            fake_validity = D(fake_imgs)
            
            g_loss = criterion(fake_validity,torch.ones_like(fake_validity).cuda())
            
            if g_loss < min_g_loss:
                min_g_loss = g_loss
                g_save_path = os.path.join(weight_path,'Generator',f'G_epoch-{epoch}-loss{g_loss:4f}.pth')    
                torch.save(G.state_dict(),g_save_path)

            g_loss.backward()
            optimizer_G.step()

            # d_losses.append(d_loss)
            # g_losses.append(g_lass)
            
            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                f"[D loss: {d_loss.item():3f}] "
                f"[G loss: {g_loss.item():3f}]")

    if epoch & save_interval == 0:
        d_save_path = os.path.join(weight_path,'Discriminator',f'D_epoch-{epoch}-{d_loss:4f}.pth')
        torch.save(D.state_dict(),d_save_path)
        g_save_path = os.path.join(weight_path,'Generator',f'G_epoch-{epoch}-loss{g_loss:4f}.pth')    
        torch.save(G.state_dict(),g_save_path)


#izi
# z = torch.randn(64,latent_dim, 1,1).cuda() # 64,100
# fake_imgs = G(z) # 64,1,28,28
# fake_z = E(fake_imgs)
# reconf_imgs = G(fake_z)
# utils.imshow_grid(reconf_imgs)

# G.load_state_dict(torch.load('./runs/Generator/G_epoch-496-loss13.220990.pth'))
# D.load_state_dict(torch.load('./runs/Discriminator/D_epoch-496-0.000005.pth'))

G.eval()
D.eval()

criterion = nn.MSELoss()

optimizer_E = torch.optim.Adam(E.parameters(), lr=lr,betas = (0.0,0.999))

padding_epoch = len(str(n_epochs))
padding_i = len(str(len(train_dataloader)))
kappa = 1.0
e_losses = []

min_loss = 999999
for epoch in range(n_epochs):
    for i, sample in enumerate(train_dataloader):
        real_imgs = sample['image'].cuda()
        
        optimizer_E.zero_grad()
        z = E(real_imgs)
        fake_imgs = G(z)

        real_features = D.forward_features(real_imgs)
        fake_features = D.forward_features(fake_imgs)

        loss_imgs = criterion(fake_imgs, real_imgs)
        loss_features = criterion(fake_features,real_features)
        e_loss = loss_imgs + kappa * loss_features
        if e_loss < min_loss:
                min_loss = e_loss
                encoder_path=os.path.join(weight_path,'Autoencoder',f'_update_min_loss_E_epoch-{epoch}.pth')
                torch.save(E.state_dict(),encoder_path)   
        
        e_loss.backward()
        optimizer_E.step()

        if i % n_critic == 0:
            e_losses.append(e_loss)
            print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                    f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                    f"[E loss: {e_loss.item():3f}]")
    encoder_path=os.path.join(weight_path,'Autoencoder',f'E_epoch-{epoch}.pth')
    torch.save(E.state_dict(),encoder_path) 

print('train_finish~~~~')