# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:22:33 2018

@author: Wook
"""

import torch.nn as nn

# Generate code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        
        # 
        self.layer = nn.Sequential(
                # input is Z, going into a convolution
                # Z size :  100
                # feature map size : 64
                nn.ConvTranspose2d(100, 64*8,4,1,0,bias = False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                
                
                # state size. (64*8) x 4 x 4
                nn.ConvTranspose2d(64*8, 64*4,4,2,1,bias = False),
                nn.BatchNorm2d(64*4),
                nn.ReLU(True),
                
                
                # state size (64*4) x 8 x 8 
                nn.ConvTranspose2d(64*4, 64*2,4,2,1,bias = False),
                nn.BatchNorm2d(64*2),
                nn.ReLU(True),
                
                # state size (64*2) x 16 x 16
                nn.ConvTranspose2d(64*2, 64, 4,2,1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                # state size (64) x 32 x 32
                nn.ConvTranspose2d(64,3,4,2,1,bias=False),
                nn.Tanh()
                
                #state size  (3) x 64 * 64
                
                
        )
    def forward(self,out):
        
        
        return self.layer(out)




if __name__ == '__main__':
    netG = Generator(1)