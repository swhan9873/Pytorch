# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:05:10 2018

@author: Wook
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 3x 32 x 32
        self.layer1 = nn.Sequential(
                
                nn.Conv2d(3,6,kernel_size=5),        # 6 x 28 x 28
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),# 6 x 14 x 14
                
                nn.Conv2d(6,16,kernel_size=5),       # 16 x 10 x 10
                nn.ReLU(),
                nn.MaxPool2d(2,2)                    # 16 x 5 x 5 
        )
        self.fc = nn.Sequential(
                nn.Linear(16*5*5,120),
                nn.Linear(120,84),
                nn.Linear(84,10)
        )
    def forward(self,x):
        out = self.layer1(x)
        print(out.shape)
        out = out.view(out.shape[0],-1)
        print(out.shape)
        out = self.fc(out)
        return out
#class LeNet(nn.Module):
#    def __init__(self):
#        super(LeNet, self).__init__()
#        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
#        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(16280, 1000)
#        self.fc2 = nn.Linear(1000, 30)
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(x.size(0), -1)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x,dim=1)
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(3,32),
                nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
                nn.Linear(32,3),
                nn.ReLU(True)
        )
    def forward(self,x):
        x = self.layer1(x)
        print('1',x.shape)
        x = self.layer2(x)
        print('2',x.shape)
        return x
def test():
    
#    model = LeNet()
#   
#    inputs = torch.randn([1,3,32,32])
#    out = model(inputs)
#    print(out.shape)
    
    model = TestNet()
    tmp =  torch.randn([128,3])
    out = model(tmp)
    
    print(out.shape)

if __name__=='__main__':
    test()        
