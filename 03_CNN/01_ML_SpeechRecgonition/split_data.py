# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:13:53 2018

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
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()

print('Number of labels: ' + str(len(dirs)))


to_keep = 'yes no up down left right on off stop go'.split()
dirs = [d for d in dirs if d in to_keep]
print(dirs)

temp = []
current_path ='./'
valid_fold = os.path.join(current_path,'val')
test_fold = os.path.join(current_path,'ts')

if not os.path.exists(valid_fold):
    os.mkdir(valid_fold)
if not os.path.exists(test_fold):
    os.mkdir(test_fold)


for direct in dirs:
    print('direct',direct)
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
  
    train_set, test_set = train_test_split(waves,   test_size = 0.2, random_state=42)
    valid_set, test_set = train_test_split(test_set,test_size = 0.5, random_state=42)
    
    for file in valid_set:
        file_path = join(train_audio_path, direct,file)
        
        if file.endswith('.wav'):
            val_path = os.path.join(valid_fold,direct)
            if not os.path.exists(val_path):
                os.mkdir(val_path)
            shutil.move(file_path,val_path)
    
    for file in test_set:
        file_path = join(train_audio_path, direct,file)
        
        if file.endswith('.wav'):
            ts_path = os.path.join(test_fold,direct)
            if not os.path.exists(ts_path):
                os.mkdir(ts_path)
            shutil.move(file_path,ts_path)
    
    

    
    
    
    
    