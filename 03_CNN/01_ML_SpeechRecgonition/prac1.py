# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:00:58 2018

@author: gist
"""

import tensorflow as tf
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import IPython.display
from tensorflow.python.platform import gfile
import numpy as np

def get_mfcc(wave_path,PAD_WIDTH= 44):
    wave,sr = librosa.load(wave_path,mono=True)
    mfccs = librosa.feature.mfcc(y=wave,sr=sr,n_mfcc=20)
    mfccs = np.pad(mfccs,((0,0),(0,PAD_WIDTH - len(mfccs[0]))),mode = 'constant')
    return mfccs

path_train = './tr/audio/'

# Example audio file from datasets
path = os.path.join(path_train,'yes','*.wav')
wave_path = gfile.Glob(path)[80]
print(wave_path)
IPython.display.Audio(wave_path)

# Visualizing Audio Wave
wave,sr = librosa.load(wave_path,mono=True)
plt.figure(figsize=(12,4))
librosa.display.waveplot(wave,sr=sr)

# Visualizing Mel Freqeuncy Cepstal Coefficients
mfccs = librosa.feature.mfcc(y = wave,sr= sr,n_mfcc =20)
plt.figure(figsize = (12,4))
librosa.display.specshow(mfccs,x_axis='time')




test_mfcc = get_mfcc(wave_path)
#print(test_mfcc)
plt.figure(figsize = (12,4))
plt.title('test_MFCC')
librosa.display.specshow(test_mfcc,x_axis='time')
plt.colorbar()


D = np.abs(librosa.stft(wave))**2
spec = librosa.feature.melspectrogram(y=wave,sr=sr,n_mels=128,fmax=8000)
plt.figure(figsize = (12,4))
plt.title('melspectrogram')
librosa.display.specshow(librosa.power_to_db(spec,ref=np.max),x_axis='time')
plt.colorbar()


# Visualizing 2D feature
#

# Visualizing label encoding
#label_to_index_map = {'yes':0 , 'no':1, 'bed':2}
#_,label = os.path.split(os.path.dirname(wave_path))
#print('_ is :', _)
#encoding = [0] * len(label_to_index_map)
#encoding[label_to_index_map[label]] = 1 
#print('wave path:',wave_path)
#print('label:',label)
#print('encoding:',encoding)
