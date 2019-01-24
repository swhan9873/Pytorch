# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:37:44 2019

@author: Wook
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# device configuration
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_GPU else "cpu");print(device)

# create a directory if not exist
if not os.path.exists('./cvae_img'):
    os.mkdir('./cvae_img')


# build the CVAE architecture
class Encoder(nn.Module):
    def __init__(self, h_dim=800, z_dim=100):
        super(Encoder,self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(1,8,3,padding=1),     # batch x 8 x 28 x 28
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2,2),              # batch x 8 x 14 x 14
                
                nn.Conv2d(8,16,3,padding=1),    # batch x 16 x 14 x 14
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                
                nn.Conv2d(16,32,3,padding=1),   # batch x 32 x 7 x7
                nn.ReLU()
        )
        self.encoder_mu = nn.Sequential(
                nn.Linear(32*7*7,h_dim),
                nn.Linear(h_dim,z_dim)
        )
        self.encoder_logvar = nn.Sequential(
                nn.Linear(32*7*7,h_dim),
                nn.Linear(h_dim,z_dim)
                
        )
    def reparameterize(self,mu,logvar):
        # log variance = log std**2
        std     = torch.exp(logvar/2)
        
        # eps ~ N(0,1) -> use libiray : torch.randn_like libiray 
        epsilon = torch.randn_like(std)
        
        # reparameterize trick
        z = mu + std*epsilon
        return z
    
    def forward(self,x):
        
        out     = self.encoder(x)
        # to fit the dimension , input data flatten
        # [batch,32,7,7] -> [batch,32*7*7]
        out     = out.view(out.size(0),-1)
        mu      = self.encoder_mu(out)
        logvar  = self.encoder_logvar(out)
        
        # latnet variable
        z = self.reparameterize(mu,logvar)
        
        # z.shape batch x 100 
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, h_dim=800, z_dim=100):
        super(Decoder,self).__init__()
        
        self.decoder_z = nn.Sequential(
                nn.Linear(z_dim,h_dim),
                nn.ReLU(),
                nn.Linear(h_dim,32*7*7),
                nn.ReLU()
        )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32,16,3,2,1,1), # batch x 16 x 14 x 14
                nn.ReLU(),
                nn.BatchNorm2d(16),
                
                nn.ConvTranspose2d(16,8,3,2,1,1),  # batch x 8 x 28 x28
                nn.ReLU(),
                nn.BatchNorm2d(8),
                
                nn.ConvTranspose2d(8,1,3,1,1),     # batch x 1 x 28 x28
                nn.BatchNorm2d(1)
        )
        
    def forward(self,z):
        
        out = self.decoder_z(z)
        # [batch, 32*7*7] -> [batch,32,7,7]
        out = out.view(out.size(0),32,7,7)
        
        # [batch,1,28,28]
        out = self.decoder(out)
        # print(out.shape)
        recon_x = torch.sigmoid(out)
        
        return recon_x
        
    

class CVAE(nn.Module):
    def __init__(self,encoder,decoder):
        super(CVAE,self).__init__()
        
        self.Encoder = encoder
        self.Decoder = decoder
        
    def forward(self,x):
        z, mu, logvar = self.Encoder(x)
        recon_x = self.Decoder(z)
        
        return recon_x ,mu, logvar
    
    
def main():
    
    # initialize hyperparameter
    batch_size      = 128
    learning_late   = 0.0005
    num_epoch       = 100
    test_batch_size = 100
    
    # image augmentation
    img_transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # load the train data
    trainset = MNIST(root='./data',train=True,transform= img_transform)
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,num_workers=4)
    
    # load the test data
    testset = MNIST(root='./data',train=False,transform = img_transform)
    testloader = DataLoader(dataset=testset,batch_size=test_batch_size,shuffle=False,num_workers=4)
    
    
    # define the models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = CVAE(encoder,decoder).to(device)
    
    # define the loss function and optimizer with weight decay using L2 Norm
    reconstruct_func = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(),lr=learning_late,weight_decay=1e-5)
    
    test_loss_list = []
    best_test_loss = np.inf
    iteration = 0
    limit = 4
    early_stopping = False
    
    # start training with ealry stopping !
    for epoch in range(num_epoch):
        
        print('--'*19)
        # training
        train(trainloader,model,reconstruct_func,optimizer,epoch) 
        test_loss = test(testloader,model,reconstruct_func,optimizer,epoch)
        
        # test loss saved
        test_loss_list.append(test_loss)
        
        
        # save check point
        if test_loss > best_test_loss:
            iteration += 1
            print('Loss does not decreased.. ,iteration {0}'.format(str(iteration)))
        else:
            
            iteration = 0
            best_test_loss = test_loss
            print('saving model')
            
            state = {
                'net'   :model.state_dict(),
                'loss'  :test_loss,
                'epoch' :epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/wook_0124.pth')
        
        # early stopping
        if iteration == limit:
            early_stopping = True
            stopping_epoch = epoch+1
            print('Early Stopping !')
            break            
    print('Finished Training')
    
    # same the dimension
    size = np.arange(stopping_epoch) if early_stopping else np.arange(num_epoch)
    
    # show the test loss
    plt.plot(size,test_loss_list)
    plt.title('Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('test_loss')
    plt.show()    


def train(trainloader,model,reconstruct_func,optimizer,epoch):
    
    epoch_loss = 0
    train_loss = 0
    
    model.train()
    # training 
    for batch_idx, (x,_) in enumerate(trainloader,0):
        
        x = x.to(device)
        
        # zero the parameter graidents
        optimizer.zero_grad()
        
        # ~~ ing
        recon_x, mu , logvar = model(x)
        
        # compute reconstruction loss and KL divergence
        Reconstruct_error = reconstruct_func(recon_x,x)
        Regularization  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ELBO = Reconstruct_error + Regularization
        
        # backprop and optimize
        loss = ELBO
        loss.backward()
        optimizer.step()
        
        # statistics
        epoch_loss += loss.item()
     
    train_loss = epoch_loss / len(trainloader.dataset)
    print("Epoch[{}/{}], Train Loss : {:.4f}".format(epoch+1,100,train_loss))
    
    # cuda memory RESET
    if USE_GPU:
        torch.cuda.empty_cache()

def test(testloader,model,reconstruct_func,optimizer,epoch):
    
    epoch_loss = 0
    test_loss  = 0
    
    model.eval()
    
    # eval
    with torch.no_grad():
        for batch_idx, (x,_) in enumerate(testloader,0):
            
            x = x.to(device)
            
            recon_x,mu,logvar = model(x)
        
            # compute reconstruction loss and KL divergence
            Reconstruct_error = reconstruct_func(recon_x,x)
            Regularization  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            ELBO = Reconstruct_error + Regularization
            
            loss = ELBO
            epoch_loss += loss.item()
            
        test_loss = epoch_loss / len(testloader.dataset)
        print("Epoch[{}/{}], TEST Loss : {:.4f}".format(epoch+1,100,test_loss))
        
        
        if epoch % 10 == 0:
            with torch.no_grad():
                x_concat = torch.cat([x.view(-1, 1, 28, 28), recon_x.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat,'./cvae_img/reconst-{}.png'.format(epoch))
        return test_loss
        
def check():
    
    encoder = Encoder()
    # [batch,channel,width,height]
    x = torch.randn([128,1,28,28])
    z,mu,logvar = encoder(x)
#    print(z,mu,logvar)
#    print('z shape',z.shape)
    
    decoder = Decoder()
    recon_x = decoder(z)
    
    
    cvae = CVAE(encoder,decoder)
    x1 = torch.randn([128,1,28,28])
    recon_x1 = cvae(x1)
    print('222',recon_x1.shape)
    

def to_img(x):
    x = 0.5* (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

if __name__ == '__main__':
    #check()
    
    main()
    