# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:30:55 2018

@author: gist
"""
from __future__ import print_function
# Change this to True to replicate the result
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import shutil


train_audio_path = './tr/audio/'
validation_path = './val/'
test_path = './ts/'
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()

print('Number of labels: ' + str(len(dirs)))


to_keep = 'yes no up down left right on off stop go'.split()
dirs = [d for d in dirs if d in to_keep]
print(dirs)


# Check Train data 
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    print('Train :',direct,len(waves))
    
    
# Check Val data 
for direct in dirs:
    waves = [f for f in os.listdir(join(validation_path, direct)) if f.endswith('.wav')]
    print('Val :',direct,len(waves))
    
    
# Check Test data 
for direct in dirs:
    waves = [f for f in os.listdir(join(test_path, direct)) if f.endswith('.wav')]
    print('Test :',direct,len(waves))