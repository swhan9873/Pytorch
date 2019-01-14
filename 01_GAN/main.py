# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:52:31 2018

@author: Wook
"""

from __future__ import print_function
#%matplotlib inline

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as andimation
from torch.utils.data import DataLoader
from IPython.display import HTML
from model import Generator

# check the gpu or cpu
use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu');print(device)

def main():
    # set random seem for reproducibility
    manual_seed = 999
    print('Random Seed: ',manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    
    # root directory for dataset
    dataroot = './data/celeba'
    
    # number of workers for dataloader
    workers = 2
    
    # batch size during traing
    batch_size = 128
    
    # spatial size of training image. All images will be resized to this
    # size using a transformer 
    # default size : 64 x 64
    image_size = 64
    
    
    # number of channel in the training image. for color image this is 3
    nc = 3
    
    # size of z latent vector
    # i.e size of generator input
    nz  = 100
    
    # size of feature map in generator
    ngf = 64
    
    # size of feature map in discriminator
    ndf = 64
    
    # number of training epoch
    num_epochs = 5
    
    # learning rate for optimizers
    lr = 0.0002
    
    # beta1 hyperparmeter for Adam optimizer
    beta1= 0.5
    
    # number of GPUs avaliable. use 0 for CPU mode
    ngpu  = 1 if use_gpu else 0
    

    
    
    data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    
    # create the dataset
    dataset = datasets.ImageFolder(root= dataroot,
                                   transform = data_transforms)

    # create the dataloader
    dataloader = DataLoader(dataset,batch_size=batch_size,
                            shuffle=True,num_workers=workers)

    # display the image
    imgshow(dataloader)
    
    
    
    # create the generator
    netG = Generator(ngpu).to(device)
    
    # handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    
    # apply the weight_init function to randomly initalize all weights
    # mean = 0, stdev = 0.2
    
    netG.apply(weights_init)
    
    print(netG)
    

# custom weights initialization called on netG and netD    
def weights_init(m):
    classname = m.__class__.__name__
    with torch.no_grad():
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
    
    


def imgshow(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('traing images')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
                                             padding=2, normalize=True).cpu(),(1,2,0)))
    

if __name__ == '__main__':
    main()