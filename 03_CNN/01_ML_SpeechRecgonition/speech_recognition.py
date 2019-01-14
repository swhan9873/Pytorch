# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:12:58 2018

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
import random
import time
import pandas as pd

PATH_TRAIN = './tr/audio'
PATH_TEST ='./ts/'
PATH_VAL = './val/'
BATCH_SIZE = 100
num_epoch = 500
ITERATION_VAL = 20
ITERATION_TEST = 10
EVAL_EVERY = 5
HEIGHT = 20
WIDTH = 44
NUM_LABELS = 0
LEARNING_RATE = 0.0001
LOGDIR = './log/'
TEST_LOGDIR = './log_test/'

LABEL_TO_INDEX_MAP = {}


# Create a mapping from labels to index and vice versa
def init(path):
    labels = os.listdir(path)
    index = 0
    for label in labels:
        LABEL_TO_INDEX_MAP[label] = index
        index += 1
    
    global NUM_LABELS
    NUM_LABELS = len(LABEL_TO_INDEX_MAP)
    
# Function to convert word label to vector
def one_hot_encoding(label):
    #encoding = [0]*NUM_LABELS
    encoding = [0] * len(LABEL_TO_INDEX_MAP)
    encoding[LABEL_TO_INDEX_MAP[label]] = 1
    return encoding

# Function to get Mel Frequency Cepstrum coefficients
def get_mfcc(wave_path,PAD_WIDTH= WIDTH):
    wave,sr = librosa.load(wave_path,mono=True)
    mfccs = librosa.feature.mfcc(y=wave,sr=sr,n_mfcc=HEIGHT)
    mfccs = np.pad(mfccs,((0,0),(0,PAD_WIDTH - len(mfccs[0]))),mode = 'constant')
    return mfccs

# Function to get net batch of labels and audio feature
def get_batch(batch_size,path):
    X = []
    Y = []
    random.seed(5996)
    path = os.path.join(path,'*','*.wav')
    waves = gfile.Glob(path)
    while True:
        random.shuffle(waves)
        for wave_path in waves:
            _,label= os.path.split(os.path.dirname(wave_path))
            X.append(get_mfcc(wave_path))
            Y.append(one_hot_encoding(label))
            
            if len(X) == batch_size:
                yield X,Y
                X = []
                Y = []
    
def model(input_data,dropout):
    # Frist Convolution layer
    with tf.name_scope('Conv1'):
        input_4D = tf.reshape(input_data,[-1,HEIGHT,WIDTH,1])
        w1 = tf.Variable(tf.truncated_normal([12,8,1,44],stddev=0.01),name='w')
        b1 = tf.Variable(tf.zeros([44]),name='b')
        conv1 = tf.nn.conv2d(input_4D,w1,strides=[1,1,1,1],padding='SAME')
        act1 = tf.nn.relu(conv1 + b1)
        drop1 = tf.nn.dropout(act1,dropout)
        max_pool1 = tf.nn.max_pool(drop1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        tf.summary.histogram('weight',w1)
        tf.summary.histogram('bias',b1)
        tf.summary.histogram('activation',act1)
        tf.summary.histogram('dropouts ',drop1)
    
    
    # Second Convolution layer
    with tf.name_scope('Conv2'):
        w2 = tf.Variable(tf.truncated_normal([6,4,44,44],stddev=0.01),name='w')
        b2 = tf.Variable(tf.zeros([44]),name='b')
        conv2 = tf.nn.conv2d(max_pool1,w2,strides=[1,1,1,1],padding='SAME')
        act2 = tf.nn.relu(conv2 +  b2)
        drop2 = tf.nn.dropout(act2,dropout)
        tf.summary.histogram('weight',w2)
        tf.summary.histogram('bias',b2)
        tf.summary.histogram('activation',act2)
        tf.summary.histogram('dropout',drop2)
    
    # Reshaping for Fully connected layer
        conv_shape = drop2.get_shape()
        count = int(conv_shape[1]*conv_shape[2]*conv_shape[3])
        flat_output = tf.reshape(drop2,[-1,count])
        #print('flat:',flat_output)
        
    with tf.name_scope('FC'):
        w3 = tf.Variable(tf.truncated_normal([count,NUM_LABELS],stddev=0.01))
        b3 = tf.Variable(tf.zeros([NUM_LABELS]))
        fc = tf.add(tf.matmul(flat_output,w3),b3)
        tf.summary.histogram('weight',w3)
        tf.summary.histogram('bias',b3)
    

    return fc
    
    
def main():
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    # Placeholder for input
    x = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH], name='input')
 
    
    # Placeholder for label
    y = tf.placeholder(tf.float32,shape =[None,NUM_LABELS],name='label')
    
  
    dropout = tf.placeholder(dtype=tf.float32,name='dropout')
    # NN model
    
    net = model(x,dropout)
    
#    # loss function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net,labels=y))
        tf.summary.scalar('loss',loss)
                                 
    # Optimizer for loss function
    with tf.name_scope('train'):
        trian_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
    
    # Accuracy function
    with tf.name_scope('accuracy'):
        predicted= tf.argmax(net,1)
        truth = tf.argmax(y,1)
        correct_prediction = tf.equal(predicted,truth)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        confusion_matrix = tf.confusion_matrix(truth,predicted,num_classes = NUM_LABELS)
        tf.summary.scalar('accuracy',accuracy)
        
    # Setting tensor board
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(TEST_LOGDIR)
    
    
    #Training model
    print('Start training\n')

    batch = get_batch(BATCH_SIZE,PATH_TRAIN)
    start_time = time.time()
    for i in range(1,num_epoch+1):
        X,Y = next(batch)
        if i % EVAL_EVERY == 0:
            [train_accuracy, train_loss,s] = sess.run([accuracy,loss,summ],feed_dict={x:X,y:Y,dropout:1.0})
            acc_and_loss = [i,train_loss,train_accuracy*100]
            print('Iteration # {}. Train loss: {:.2f}. Train Acc: {:.0f}%'.format(*acc_and_loss))
            writer.add_summary(s,i)
        if i % (EVAL_EVERY*20) == 0:
            train_confusion_matrix = sess.run([confusion_matrix],feed_dict = {x:X,y:Y,dropout:1.0})
            header = LABEL_TO_INDEX_MAP.keys()
            df = pd.DataFrame(np.reshape(train_confusion_matrix,(NUM_LABELS, NUM_LABELS)),index=header)
            print('\nConfusion Matrix:\n {}\n'.format(df))
            saver.save(sess,os.path.join(LOGDIR,"model.ckpt"),i)
        
        sess.run(trian_step,feed_dict={x:X,y:Y,dropout:0.7})
        
    print('\nTotal training time : {:0f} seconds'.format(time.time() -start_time))
    
    
    # Valdation model
    print('Start validation \n')
    batch = get_batch(BATCH_SIZE,PATH_VAL)
    total_acc = 0
    for i in range(ITERATION_VAL):
        X,Y = next(batch,PATH_VAL)
        val_acc,s = sess.run([accuracy,summ],feed_dict= {x:X,y:Y,dropout:1.0})
        print('Iteration # {}.  Val Acc: {:.0f}%'.format(i+1,val_acc*100))
        total_acc += (val_acc /ITERATION_VAL)
        test_writer.add_summary(s,i)
    
    print('\nFinal Val Accuracy: {:.0f}%'.format(total_acc*100))
     
    
    # Testing model
    print('Start testing \n')
    batch = get_batch(BATCH_SIZE,PATH_TEST)
    total_acc = 0
    for i in range(ITERATION_TEST):
        X,Y = next(batch,PATH_TEST)
        test_acc,s = sess.run([accuracy,summ],feed_dict= {x:X,y:Y,dropout:1.0})
        print('Iteration # {}.  Test Acc: {:.0f}%'.format(i+1,test_acc*100))
        total_acc += (test_acc /ITERATION_TEST)
        test_writer.add_summary(s,i)
    
    print('\nFinal Test Accuracy: {:.0f}%'.format(total_acc*100))
if __name__ == '__main__':
    init(PATH_TRAIN)
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    