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
import matplotlib.pyplot as plt
import numpy as np
from model import MyLeNet, MyLeNet2 ,VGG
from data_loader import SpectrumDataset, MelSpectrumDatasets
import time


"""  Pytorch Speech Recognition """
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

def main():
    
    # initialize the hyperparmeter
    num_epoch = 30
    train_batch_size = 128
    val_batch_size = 100
    test_batch_size = 100
    learing_late = 0.001
    
    # setting path
    train_path = './tr/audio'
    valid_path = './val'
    test_path = './ts'
    
    # STFT parameter
    window_size = .02
    window_stride = .01
    window_type='hamming'
    
    # Mel Spectrum parameter
    n_mels =128
    fmax = 8000
    max_len = 50
    
    """ Spectrum datasets"""
    
    # load the train data
    trainset = SpectrumDataset(train_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                             normalize=True)
    trainloader = DataLoader(trainset,batch_size=train_batch_size,shuffle=True,num_workers=0)

    # load the validation data
    validset = SpectrumDataset(valid_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                              normalize=True)
    validloader = DataLoader(validset,batch_size=val_batch_size,shuffle=False,num_workers=0)
    

    
    # load the test dada
    testset = SpectrumDataset(test_path,window_size=window_size,window_stride=window_stride,window_type=window_type,
                              normalize=True)
    testloader = DataLoader(testset,batch_size=test_batch_size,shuffle=False,num_workers=0)
    
    
    """ Mel-Spectrum datasets"""
    
#    # load the train data
#    trainset = MelSpectrumDatasets(train_path,n_mels = n_mels,fmax = fmax, max_len= max_len,
#                             normalize=True)
#    trainloader = DataLoader(trainset,batch_size=train_batch_size,shuffle=True,num_workers=0)
#    
#    # load the validation data
#    validset = MelSpectrumDatasets(valid_path,n_mels = n_mels,fmax = fmax, max_len= max_len,
#                             normalize=True)
#    validloader = DataLoader(validset,batch_size=val_batch_size,shuffle=False,num_workers=0)
#    
#    # load the test dada
#    testset = MelSpectrumDatasets(test_path,n_mels = n_mels,fmax = fmax, max_len= max_len,
#                             normalize=True)
#    testloader = DataLoader(testset,batch_size=test_batch_size,shuffle=False,num_workers=0)
    
    
    # model 
    # spectrogram --> MyLeNet()
    # mel-spectrogram --> MyLeNet2()
    
    net = MyLeNet()
    #net = MyLeNet2()
    if use_gpu:
        print('GPU Processing')
        net = MyLeNet().to(device)
        cudnn.benchmark = True
    
    # define optimizer with weight decay using L2 Norm
    optimizer = optim.Adam(net.parameters(),lr =learing_late,weight_decay=5e-4)
    
    
    
    # option parameter
    best_valid_loss= np.inf
    iteration = 0
    limit = 5
    val_acc_list=[]
    val_loss_list =[]
    ealry_stoping = False
    
    
    # start the train with ealry stopping !
    start_time = time.time()
    
    for epoch in range(num_epoch):
        print('\nEpoch {}/{}'.format(epoch, num_epoch))
        print('--' * 18)
        #training
        train(trainloader,net,optimizer)
        valid_loss, valid_acc = validation(validloader,net,optimizer)
        
        # loss, val saved
        val_acc_list.append(valid_acc)
        val_loss_list.append(valid_loss)
        
        #save check point
        if valid_loss > best_valid_loss:
            iteration += 1
            print('Loss does not decreased.. ,iteration {0}'.format(str(iteration)))
        else:
            
            iteration = 0
            best_valid_loss = valid_loss
            print('Saving model')
            
            state = {
                'net'  : net.state_dict(),
                'loss' : valid_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/wook_1209.pth')
        
        # early stopping
        if iteration == limit:
            ealry_stoping = True
            stopping_epoch = epoch+1
            print('Early Stopping !')
            break
        
    print('Finished Training')
    
    
    # same the dimension
    size = np.arange(stopping_epoch) if ealry_stoping else np.arange(num_epoch)
    
    # show the valdation accuracy
    plt.plot(size,val_acc_list)
    plt.title('Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val_acc')
    plt.show()
    
    # show the valdation loss
    plt.plot(size,val_loss_list)
    plt.title('Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.show()
    
    print('\nTotal training time : {:.4f} seconds'.format(time.time() -start_time))
    
    # predict the testdata  
    Accuracy = test(testloader,net,optimizer)
    
    print('\nTest Acc: {:.3f}'.format(Accuracy))
    
def train(trainloader,net,optimizer):
    
    # train !
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    net.train()
    for i , (inputs,labels) in enumerate(trainloader,0):
    
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
        # zero the parameter gradients    
        optimizer.zero_grad()
        
        # predicted
        outputs = net(inputs)
        _,predicted = torch.max(outputs.data,1)
        loss =  F.nll_loss(outputs,labels)
        
        # backward + optimizer
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
        running_corrects += predicted.eq(labels.data).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100 *running_corrects / total
     
    print('Train Loss: {:.3f} Train Acc: {:.3f}'.format(epoch_loss, epoch_acc))
    
    # cuda memory reset.
    if use_gpu:
        torch.cuda.empty_cache()
    
def validation(validloader,net,optimizer):
    
    # validatoin !
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validloader,0):
            if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
            # predicted
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            loss =  F.nll_loss(outputs,labels)
          
            
            # statistics
            running_loss += loss.item()
            running_corrects += predicted.eq(labels.data).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(validloader)
        epoch_acc = 100 * running_corrects / total
     
        print('Valid Loss: {:.3f} Valid Acc: {:.3f}'.format(epoch_loss, epoch_acc))
    
    return epoch_loss,epoch_acc

def test(testloader,net,optimizer):
    
    # evaluation !
    test_loss = 0.0
    test_corrects = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader,0):
            if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
            # predicted
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            loss =  F.nll_loss(outputs,labels)
        
            # statistics
            test_loss += loss.item()
            test_corrects += predicted.eq(labels.data).sum().item()
            total += labels.size(0)
            
        test_acc = 100 * test_corrects / total
   
    return test_acc     
        
if __name__ == '__main__':
    main()