# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:04:24 2018

@author: Wook
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from model import LeNet,VGG
from data_loader import SpeechDataset


"""  Pytorch Speech Recognition """
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

def main():
    
    #initialization hyperparmeter
    num_epoch = 200
    train_batch_size = 128
    val_batch_size = 100
    test_batch_size = 100
    learing_late = 0.001
    
    #setting path
    train_path = './tr/audio'
    valid_path = './val'
    test_path = './ts'
    
    #STFT
    window_size = .02
    window_stride = .01
    window_type='hamming'
    
    

    #load the train data
    trainset = SpeechDataset(train_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                             normalize=True)
    trainloader = DataLoader(trainset,batch_size=train_batch_size,shuffle=True,num_workers=0)
    

    #load the validation data
    validset = SpeechDataset(valid_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                              normalize=True)
    validloader = DataLoader(validset,batch_size=val_batch_size,shuffle=True,num_workers=4)
    
    
    #load the test dada
    testset = SpeechDataset(test_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                              normalize=True)
    testloader = DataLoader(testset,batch_size=test_batch_size,shuffle=False,num_workers=4)
    
    
    #build model
    net = LeNet()
    if use_gpu:
        print('GPU Processing')
        net = LeNet().to(device)
        cudnn.benchmark = True
    
    #define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr =learing_late,weight_decay=5e-4)
    

    best_valid_loss= np.inf
    iteration = 0
    limit = 5
    val_acc_list=[]
    val_loss_list =[]
    
    
    #Start the train with ealry stopping !
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch))
        print('--' * 15)
        #training
        train(trainloader,net,loss_func,optimizer,epoch)
        valid_loss, valid_acc = validation(validloader,net,loss_func,optimizer,epoch)
        
        # loss, val saved
        val_acc_list.append(valid_acc)
        val_loss_list.append(valid_loss)
        
        # save check point
        if valid_loss > best_valid_loss:
            iteration += 1
            print('Loss was not decreased, iteration {0}'.format(str(iteration)))
        else:
            
            iteration = 0
            best_valid_loss = valid_loss
            print('Saving model')
            
            state = {
                'net' : net.state_dict(),
                'loss':valid_loss,
                'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/wook_1130.pth')
        
        # early stopping
        if iteration == limit:
            print('Early Stopping')
            break
        
    print('Finished Training')
    
    #show the valdation accuracy
    plt.plot(np.arange(num_epoch),val_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('val_acc')
    plt.show()
    
    #show the valdation loss
    plt.plot(np.arange(num_epoch),val_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.show()
    
    
    #Prediction !!!
    #test(testset,testloader,models,test_batch)

    
def train(trainloader,net,loss_func,optimizer,epoch):
    
    #train
    running_loss = 0.0
    running_corrects = 0
    
    net.train()
    for i , (inputs,labels) in enumerate(trainloader,0):
    
        if use_gpu:
            inputs = inputs.to(device)
            labels = inputs.to(device)
            
        # zero the parameter gradients    
        optimizer.zero_grad()
        
        #predicted
        outputs = net(inputs)
        _,predicted = torch.max(outputs.data,1)
        loss = loss_func(outputs,labels)
        
        #backward + optimizer
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predicted == labels.data)
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = running_corrects.double() / len(trainloader)
     
    print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    # cuda memory reset.
    if use_gpu:
        torch.cuda.empty_cache()
    
def validation(validloader,net,loss_func,optimizer,epoch):
    
    # test !
    running_loss = 0.0
    running_corrects = 0
    
    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validloader,0):
            if use_gpu:
                inputs = inputs.to(device)
                labels = inputs.to(device)
                
            #predicted
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            loss = loss_func(outputs,labels)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)
            
        epoch_loss = running_loss / len(validloader)
        epoch_acc = running_corrects.double() / len(validloader)
     
        print('Valid Loss: {:.4f} Valid Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    
if __name__ == '__main__':
    main()
