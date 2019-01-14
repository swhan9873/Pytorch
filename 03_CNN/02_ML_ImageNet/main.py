# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:37:24 2018

@author: Yes or Yes
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init

import os
import argparse
import csv

#########################################################################################################################

# CNN Models
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512 , 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



###########################################################################################################################

# Parse arguments
parser = argparse.ArgumentParser(description='Gauza~ ImageNet by Wook !')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

# Check the GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu");print(device)

# best valid accuracy
best_acc = 0.0


# epoch, batch_size
num_epoch = 100
train_batchsize = 128
valid_batchsize = 100
test_batchsize = 100

net = VGG('VGG13')
if use_gpu:
    print('GPU Processing')
    net.to(device)
    

def main():
    global best_acc
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
        
   # load the Training data 
    
    trainset = torchvision.datasets.ImageFolder(root='./all/tr',transform=transform_train)
    trainloader = DataLoader(trainset,batch_size = train_batchsize,shuffle=True,num_workers=4)     
    
    print(trainset.__getitem__(0)[0].size()) # 3 32 32
    print(trainset.__len__()) # 20000
    print(len(trainloader))
        
                    
    # validation data 가져오기
    validset = torchvision.datasets.ImageFolder(root='./all/val',transform=transform)
    validloader = DataLoader(validset,batch_size = valid_batchsize ,shuffle=True,num_workers=4)                
    
    
    # test data 가져오기
    testset = torchvision.datasets.ImageFolder(root='./all/ts',transform=transform)
    testloader = DataLoader(testset,batch_size = batch_size ,shuffle=False,num_workers=4)
    
    
    #cudnn.benchmark = True
    
    # define loss function (criterion) and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr= args.lr,weight_decay=5e-4)
    scheduler  = optim.lr_scheduler.StepLR(optimizer,step_size=60)
    # Train and Val
    for epoch in range(num_epoch):
        scheduler.step(epoch)
        
        # train test
        train_loss,train_acc = train(trainloader,net,loss_func,optimizer,epoch)
        # valid test
        val_loss,val_acc = validate()
        
        if val_acc > best_acc:
            best_acc = val_acc
    

    print('best val_acc :',best_acc )        

def train(trainloader,net,loss_func,optimizer,epoch):
    
    training_loss = 0
    correct = 0
    total = 0
    train_size = len(trainloader)
    
    if epoch % 10 == 0:
        print(optimizer)
    """ Trainging """
    net.train()
    for i, (inputs,labels) in enumerate(trainloader,0):
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs= net(inputs)
        
        loss = loss_func(outputs,labels)
        loss.backward()
        optimizer.step()
        
        _,predicted = torch.max(outputs.data,1)
        
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().item()
        
        #통계 출력
        training_loss += loss.item()
        if i% train_size == train_size-1: # print every 20000/batchsize = 5000 mini-batchs
            print('[Epoch : %d, %5d] loss: %.3f Acc: %.3f' %
              (epoch + 1, i + 1, training_loss / train_size, 100*correct / total))
            training_loss = 0.0
    if use_gpu:
        torch.cuda.empty_cache()
        
    return (training_loss / train_size, 100*correct / total )
        
        
def validate(validloader,net,loss_func):
    """ vaildation 해보기 """
    classes = ('cup', 'coffee', 'bed', 'tree','bird', 'chair', 'tea', 'bread', 'bicycle', 'sail')
    correct = 0
    total = 0
    valid_size = len(validloader)
    net.eval()
    validation_loss =0.0
    with torch.no_grad():
        for data in validloader:
            images,labels = data
            # GPU
            if use_gpu:
                images = images.to(device)
                labels = labels.to(device)
                
            outputs = net(images)
            loss = loss_func(outputs,labels)
            
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum().item()
            
            
            validation_loss += loss.item()
        print('valid_loss : %.3f valid_acc : %.3f %%' 
              % (validation_loss/valid_size ,100 * correct / total))
    
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in validloader:
            
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
        
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    
    
    print('Validaion Finished')  

def test():
    return 1


def weight_init(m):    
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        with torch.no_grad():
            if m.bias is not None:
                init.zeros_(m.bias)

if __name__ == '__main__':
    main()