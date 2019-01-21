# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:35:27 2019

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


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)


""" Convolutional AutoEncoder """

if not os.path.exists('./cae_img'):
    os.mkdir('./cae_img')
    
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoEncoder,self).__init__()
        
        self.Encoder = nn.Sequential(
                # input channel (batch) : 1
                # output channel : 16
                # filter(kerner size ) :3 
                
                # batch(3) * 16 * 10 * 10
                # ( N(28) + padding(2) - filter(3) / stride(3) ) + 1 = 10  
                nn.Conv2d(1,16,kernel_size=3,stride=3,padding=1), # batch ,16 ,10, 10 
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,8,kernel_size=2,stride=2), # batch * 8 *5 * 5
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2,2) # batch * 8  * 2 * 2 
        )
        self.Decoder = nn.Sequential(
                
                nn.ConvTranspose2d(8,8,kernel_size=3,stride=2), # batch * 8 *5 *5 
                nn.ReLU(True),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8,16,kernel_size=2,stride=2),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(16,1,kernel_size=3,stride=3,padding=1),
                nn.ReLU()
        )
    
    # 함수 호출하면 여기 하는거임 
    def forward(self,x):
        out = self.Encoder(x)
        out = self.Decoder(out)
            
        return out


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
        save_image(save, './cae_img/image_{}.png'.format(epoch))
        
    return train_loss
        
    
def check():
    model = Encoder()
    
    # example of mnist data size
    data = torch.randn([3,1,28,28])
   
    output = model(data)
    print(output.shape)
    
def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

if __name__ == '__main__':
    
    check()
    
    # initalization parameter
    batch_size = 128
    num_epoch = 2
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
    model = ConvolutionalAutoEncoder()
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
        
    