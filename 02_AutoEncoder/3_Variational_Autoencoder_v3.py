# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:37:19 2019

@author: Wook
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# device configuration
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu");print(device)


# create a directory if not exist
if not os.path.exists('./vae3_img'):
    os.mkdir('./vae3_img')


# initialize the hyper-parameter
image_size      = 28*28
h_dim           = 400 # hidden layer
z_dim           = 20  # latent layer
num_epoch       = 100
batch_size      = 128
learing_rate    = 0.0001


# image augmentation
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

# load the MNIST dataset
dataset = MNIST(root='./data',
                train=True,
                download=True,
                transform= img_transform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=28*28,h_dim=400,z_dim=20):
        super(VAE,self).__init__()
        
        self.encoder            = nn.Linear(image_size,h_dim)
        self.latent_mu          = nn.Linear(h_dim,z_dim)
        self.latent_logvar      = nn.Linear(h_dim,z_dim)
        self.latent_variable    = nn.Linear(z_dim,h_dim)
        self.decoder            = nn.Linear(h_dim,image_size)
        
    def encode(self,x):
        
        out = F.relu(self.encoder(x))
     
        mu = self.latent_mu(out)
        logvar = self.latent_logvar(out)
        
        return mu,logvar
    
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(logvar/2)
        epsilon = torch.randn_like(std)
           
        #latent variable
        z = mu + std * epsilon
        
        return z
    
    def decode(self,z):
        
        out = F.relu(self.latent_variable(z))
        out = torch.sigmoid(self.decoder(out))
        return out
    
    def forward(self,x):
        mu,logvar = self.encode(x)
        
        #latent variable
        z = self.reparameterize(mu,logvar)
        
        reconstruct_x = self.decode(z)
        return reconstruct_x,mu,logvar
    
    
model = VAE().to(device)
reconstruction_function = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(),lr=learing_rate,weight_decay=1e-5)


train_loss_list = []

# start training
for epoch in range(num_epoch):
    
    train_loss = 0
    
    model.train()
    for batch_idx, (x,_) in enumerate(dataloader,0):
        
        # change the input dimension
        x = x.to(device)
        x = x.view(x.size(0),-1)
        
        x_reconstruct, mu, logvar = model(x)
        
        # compute reconstruction loss and KL divergence
        #Reconstruct_error = F.binary_cross_entropy(x_reconstruct,x,reduction='sum')
        Reconstruct_error = reconstruction_function(x_reconstruct,x)
        Regularization  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = Reconstruct_error + Regularization
        
        # backprop and optimize
        loss = ELBO
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # statistics
        train_loss += loss.item()
        
    train_loss = train_loss / len(dataloader.dataset)
    print("Epoch[{}/{}], Train Loss : {:.4f}".format(epoch+1,num_epoch,train_loss))
    
    
    if epoch % 10 == 0:
        with torch.no_grad():
            # save the reconsturcted images
            out,_,_ = model(x)
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat,'./vae3_img/reconst-{}.png'.format(epoch+1))
            
    train_loss_list.append(train_loss)
    
# show the train loss    
plt.plot(np.arange(num_epoch),train_loss_list)
plt.ylabel('loss')
plt.show()
    
    
    
