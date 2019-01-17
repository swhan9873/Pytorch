# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:23:49 2019

@author: Wook

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

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

if not os.path.exists('./vae2_img'):
    os.mkdir('./vae2_img')

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(28*28,400),
                nn.ReLU()
        )
        self.latent_mu      = nn.Linear(400,20)
        self.latent_logvar  = nn.Linear(400,20)
        
        self.latent_variable  = nn.Sequential(
                nn.Linear(20,400),
                nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
                nn.Linear(400,28*28),
                nn.Sigmoid()
        )
    
    def Encode(self,x):
        out = self.encoder(x)
        
        mu  = self.latent_mu(out)
        logvar = self.latent_logvar(out)
        
        return mu, logvar
    def Reparameterization(self,mu,logvar):
        
        # log var = log std**2
        std = torch.exp(logvar / 2)
        
        # epsilon ~ N(0,1)
        epsilon = torch.randn_like(std)
        
        if use_gpu:
            epsilon = epsilon.to(device)
        
        # latent variable 
        
        z = mu + std*epsilon
        
        return z
    
    def Decode(self, z):
        out = self.latent_variable(z)
        out = self.decoder(out)
        return out
    
    def forward(self,x):
        mu, logvar = self.Encode(x)
        
        # latent variable
        z = self.Reparameterization(mu,logvar)
        reconstruct_x = self.Decode(z)
        
        return reconstruct_x,mu,logvar
    
def train(dataloader,model,reconstruction_function, optimizer,epoch):
    
    train_loss = 0
    
    model.train()
    
    for batch_idx ,(data,label) in enumerate(dataloader,0):
        if use_gpu:
            data = data.to(device)
            label = label.to(device)
        
        # To fit the same size, change the input data dimenstion
        data = data.view(data.size(0),-1)
        
        # zero the parameter graidents
        optimizer.zero_grad()
        
        reconstruct_x, mu, logvar = model(data)
        loss = loss_func(reconstruction_function,reconstruct_x,data,mu,logvar)
        
        # backward + oprimizer
        loss.backward()
        optimizer.step()
        
        # statistics
        train_loss += loss.item()
        
    
    train_loss = train_loss / len(dataloader.dataset)
     
    print('Train Loss: {:.4f}'.format(train_loss))
    print('\n')
     
    if epoch % 10 == 0:
        save = to_img(reconstruct_x.cpu().data)
        save_image(save, './vae2_img/image_{}.png'.format(epoch))
        
    return train_loss
        
        
        
def loss_func(reconstruction_function,reconstruct_x, x, mu,logvar):
    
    # reconstruct_x : generating data
    # x : original data
    # mu : latent mean
    # logvar : latent log variance

    Reconstruction_error = reconstruction_function(reconstruct_x,x)
    
    # KL ( q(z|x) || p(z) )
    # q(z|x)~ N(mu,sigma^2)
    # p(z) ~ N(0,1)
    Regularization = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
    
    ELBO = Reconstruction_error + Regularization
    
    return ELBO
            
def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

def check():
    model = VAE()
    data = torch.randn([3,1,28,28])
    
    
    # Linear 함수에 들어갈수 있도록 형변환 해주
    data = data.view(data.size(0),-1)
    
    output = model(data)
    print(output)

if __name__ == '__main__':
    
    
    # initalization parameter
    batch_size = 128
    num_epoch = 100
    learning_rate = 0.001
    
    # image transform
    img_transform = transforms.Compose([
            transforms.ToTensor()
    ])

    # load the MNIST dataset
    dataset = MNIST('./data',train=True,download=True,transform= img_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    # bulid model
    model = VAE()
    if use_gpu:
        model = VAE.to(device)
        
    # define reconstruction function and opimizer
    reconstruction_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    
    # check the model dimenstion and input dimension
    #check()


    train_loss_list = []
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch))
        print('--' * 15)
        
        
        # training
        train_loss = train(dataloader,model,reconstruction_function,optimizer,epoch)
        
        train_loss_list.append(train_loss)
        
    
    # show the train loss
    plt.plot(np.arange(num_epoch), train_loss_list)
    plt.ylabel('loss')
    plt.show()
        
    
