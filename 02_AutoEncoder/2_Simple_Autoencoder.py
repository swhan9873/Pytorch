# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:35:04 2019

@author: wook
"""

import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.nn.init as init

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        self.Encoder = nn.Sequential(
                nn.Linear(28*28,128),
                nn.ReLU(True),
                nn.Linear(128,64),
                nn.ReLU(True),
                nn.Linear(64,12),
                nn.ReLU(True),
                nn.Linear(12,3),
        )
        self.Decoder = nn.Sequential(
                nn.Linear(3,12),
                nn.ReLU(True),
                nn.Linear(12,64),
                nn.ReLU(True),
                nn.Linear(64,128),
                nn.ReLU(True),
                nn.Linear(128,28*28),
                nn.Tanh()
        )
        
    def forward(self,x):
        out = self.Encoder(x)
        out = self.Decoder(out)
        return out

def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

"""  Pytorch Speech Recognition """
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

def main():
    if not os.path.exists('./mlp_img'):
        os.mkdir('./mlp_img')
        
    num_epochs = 100
    batch_size = 128
    learing_rate = 0.0001
    
    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    dataset = MNIST('./data',train=True,download=True,transform = img_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    #bulid model 
    model = AutoEncoder()
    if use_gpu:
        print('Gpu learning')
        model = AutoEncoder().to(device)
        cudnn.benchmark = True
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learing_rate,weight_decay=1e-5)
    
    
    noise = init.normal_(torch.FloatTensor(4,1,6,6),0,0.1)
    print(noise)
    
    
    # train 
    for epoch in range(num_epochs):
        train(dataloader,model,loss_func,optimizer,epoch)
        
        
    

def train(dataloader,model,loss_func,optimizer,epoch):
    num_epochs = 100
    model.train()
    for i,(img,label) in enumerate(dataloader,0):
        if use_gpu:
            img= img.to(device)
            label = label.to(device)

        img = img.view(img.shape[0],-1)
        
        #================== forward =======================
        output = model(img)
        loss = loss_func(output,img)
        #================== backward ========================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #==================== log ==========================
        
    print('epoch [{}/{}], loss:{:.4f}'
      .format(epoch + 1, num_epochs, loss.item()))
    
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

        
    
if __name__=='__main__':
    main()