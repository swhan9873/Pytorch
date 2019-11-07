# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:17:22 2018

@author: gist
"""

import torch.utils.data as data

import os
import os.path
import torch

import librosa
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

CATEGORIES = """
bird dog eight four happy left nine off one seven six three two wow zero bed cat
down five go house marvin no on right sheila stop tree up yes
""".split()


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes = CATEGORIES
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in CATEGORIES:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)


    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


def melspect_loader(wave_path,n_mels = 128, fmax = 8000, max_len= 50,normalize=True):
    wave,sr = librosa.load(wave_path,mono=True)
    melspect = librosa.feature.melspectrogram(y=wave,sr=sr,n_mels=n_mels,fmax=fmax)
    
    if melspect.shape[1] < max_len:
        pad = np.zeros((melspect.shape[0], max_len - melspect.shape[1]))
        #print('pad',pad,pad.shape)
        melspect = np.hstack((melspect, pad))
        #print('spect',melspect,melspect.shape)
    elif melspect.shape[1] > max_len:
        melspect = melspect[:max_len, ]
        
    melspect = np.resize(melspect, (1, melspect.shape[0], melspect.shape[1]))
    melspect = torch.FloatTensor(melspect)
    
    
        # z-score normalization
    if normalize:
        mean = melspect.mean()
        std = melspect.std()
        if std != 0:
            melspect.add_(-mean)
            melspect.div_(std)
                  
    return melspect

class MelSpectrumDatasets(data.Dataset):
    def __init__(self,root,transform= None,target_transform=None,n_mels = 128,fmax=8000,max_len=50,normalize=True):
        classes, class_to_idx = find_classes(root)
        melspects = make_dataset(root, class_to_idx)
        if len(melspects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))
        
        self.root = root
        self.melspects = melspects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = melspect_loader
        self.n_mels = n_mels
        self.fmax = fmax
        self.max_len = max_len
        self.normalize = normalize
    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        
        path, target = self.melspects[index]
        melspects = self.loader(path, self.n_mels, self.fmax, self.max_len, self.normalize, )
        if self.transform is not None:
            melspects = self.transform(melspects)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return melspects, target
        
        
    def __len__(self):
        return len(self.melspects)
 
class SpectrumDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)