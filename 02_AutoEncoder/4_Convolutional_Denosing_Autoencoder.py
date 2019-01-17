# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:25:38 2019

@author: wook

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


if __name__ == '__main__':
    
    #initialization hyperparmeter
    batch_size = 128
    learning_rate = 0.0001
    num_epoch = 50
    
    