# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:37:44 2019

@author: wook
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

if not os.path.exists('./cae2_img'):
    os.mkdir('./cae2_img')
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1), # batch x 16 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,3,padding=1),# batch x 32 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,64,3,padding=1),# batch x 64 x 28 x 28
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2,2) # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                nn.Conv2d(64,128,3,padding =1), # b x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128,256,3,padding=1), # b x 256 x 7 x 7
                nn.ReLU()
        )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(256,128,3,2,1,1), # batch x 128 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128,64,3,1,1), # bactch x 64x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(64,16,3,1,1), # batch x 16 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(16,1,3,2,1,1), # batch x 1 x 28 x 28
                nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(x.size(0),256,7,7) # batch x 256 x 7 x7
        out = self.layer1(out)
        #print('layer1 ',out.shape)
        out = self.layer2(out)
        #print('layer2',out.shape)
        
        return out
    
    
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(ConvolutionalAutoEncoder,self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        
    def forward(self,x):
        z = self.Encoder(x)
        recon_x = self.Decoder(z)
        return recon_x
    
    
def train(dataloader,model,loss_func,optimizer,epoch):
    
    train_loss = 0
    
    model.train()
    for batch_idx, (data,label) in enumerate(dataloader,0):
        
        if use_gpu:
            data    = data.to(device)
            label   = label.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_func(output,data) 
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss = train_loss / len(dataloader.dataset)
     
    print('Train Loss: {:.4f}'.format(train_loss))
    print('\n')
     
    if epoch % 10 == 0:
        save = to_img(output.cpu().data)
        save_image(save, './cae2_img/image_{}.png'.format(epoch))
        
    return train_loss
        
    
def check():
    encoder = Encoder()
    
    # example of mnist data size
    data = torch.randn([3,1,28,28])
   
    output = encoder(data)
    
    decoder = Decoder()
    recon_data = decoder(output)
    
    vae = ConvolutionalAutoEncoder(encoder,decoder)
    recon_data2 = vae(data)

    print('recon',recon_data.shape)
    print('recon2',recon_data2.shape)
    
    print("\n")
    
    print(recon_data)    
    print(recon_data2)
    
    sys.exit()
    
    
def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

if __name__ == '__main__':
    
    #check()
    
    # initalization parameter
    batch_size = 128
    num_epoch = 100
    learning_rate = 0.001
    
    # image augmentation
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load the MNIST dataset
    dataset = MNIST('./data',train=True,download=True,transform= img_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    

    
    # bulid model
    encoder = Encoder()
    decoder = Decoder()
    
    model = ConvolutionalAutoEncoder(encoder,decoder)
    if use_gpu:
        model = ConvolutionalAutoEncoder.to(device)
        
    # define the loss function and optimizer
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    
    
    train_loss_list = []
    
    # train 
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch))
        print('--' * 15)

        # training
        train_loss = train(dataloader,model,loss_func,optimizer,epoch)
        
        train_loss_list.append(train_loss)
        
    # show the train loss
    plt.plot(np.arange(num_epoch), train_loss_list)
    plt.ylabel('loss')
    plt.show()
    
    
    
    
    
    
    
    
    