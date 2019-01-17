# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:35:42 2019

@author: wook
"""

""" Variational AutoEncoder ! """

import os
import torch
import torchvision
from torch import nn,optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn.functional as F

"""  Pytorch Speech Recognition """
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        
        self.Encoder = nn.Linear(28*28, 400)
        self.fc1 = nn.Linear(400,20) # mu
        self.fc2 = nn.Linear(400,20) # stv
        
        self.fc3 = nn.Linear(20,400) # latent variavle z
        self.Decoder = nn.Linear(400,28*28)
        
    def encode(self,x):
        h1 = F.relu(self.Encoder(x))
        
        mu = self.fc1(h1)
        # 여기서 왜 log variance (표준편차를 주는지 이해)
        logvar = self.fc2(h1)
        
        return mu, logvar
    
    
    def reparametrize(self,mu,logvar):
        
        # 0.5를 곱하는 이유는 표준편차 variance가 log를 만나면 제곱이
        # 아래로 내려오면서 2* log std 가 됨
        # 따라서 1/2를 곱해주고 다시 exp 취해주면
        # std 만 남음
        std = torch.exp(0.5*logvar)
        #std = logvar.mul(0.5).exp_()
        
        """
        torch.randn_like(input) : 
        #Epsilon eps ~ N(0,1)
        Returns a tensor with the same size as input that is filled with random numbers 
        from a normal distribution with mean 0 and variance 1
        
        """
        epsilon = torch.randn_like(std)
        if use_gpu:
            epsilon = epsilon.to(device)
        #z1 = eps.mul(std).add_(mu)
        z = mu + std*epsilon
        return z
    
    def decode(self,z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.Decoder(h3))
    
    
    def forward(self,x):
        # 28*28 이미지를 flat 하게 펼쳐줘야함. Linear에 넣기 위해서
        #out = x.view(x.shape[0],-1)
        mu, logvar = self.encode(x)
        #latent variable
        z = self.reparametrize(mu,logvar)
    
        return self.decode(z), mu, logvar
        

def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

    

def main():
    
    # initalize parameters
    num_epoch = 100
    batch_size = 128
    learning_rate = 1e-3
    
    img_transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    # load the dataset
    dataset = MNIST('./data',train=True,download=True,transform = img_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # build model
    model = VAE()
    if use_gpu:
        print('GPU processing')
        model = VAE().to(device)
    
    # 1 chanel(흑백) x 28*28 image 가 10개 있다.
#    tmp = torch.randn([10,1,28,28])
#    output = model(tmp)

    reconstruction_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    
    for epoch in range(num_epoch):
        
        train(dataloader,model,reconstruction_function,optimizer,epoch)
        
        
def loss_function(reconstruction_function,recon_x,x,mu,logvar):
    
    """
    recon_X : generating images
    x : original images
    mu = latent mean
    logvar = laternt log variance
    
    """
    
    # logvar 가 있어야 계산 할때 깔끔하게 할수있
    
    BCE = reconstruction_function(recon_x,x) # binary cross entropy
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE+KLD    
    

def train(dataloader,model,reconstruction_function,optimizer,epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (img, label) in enumerate(dataloader,0):
        img = img.view(img.shape[0],-1)
        if use_gpu:
            img = img.to(device)
            label = label.to(device)
        
        optimizer.zero_grad()
        
        recon_x,mu,logvar = model(img)
        
        loss = loss_function(reconstruction_function,recon_x,img,mu,logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader.dataset)

    print('epoch [{}/{}], loss: {:.4f}'
      .format(epoch + 1, 100, train_loss))
    
    if epoch % 10 == 0:
        save = to_img(recon_x.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))
        
    
if __name__=='__main__':
    main()