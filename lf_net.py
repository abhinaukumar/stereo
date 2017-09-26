# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:14:42 2017

@author: nownow
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Merge, Reshape, BatchNormalization
from keras.models import Model
import numpy as np

import cv2
import os

def generate_training_data(train_dir = '/home/nownow/Documents/projects/stereo/data/training/'):
    
    max_disp = 256
    
    #w = int(np.random.normal(600,60))
    #h = int(np.random.normal(150,30))
    
    l_dir = train_dir+'image_2/'
    r_dir = train_dir+'image_3/'
    d_dir = train_dir+'disp_noc_0/'
    
    l_f = sorted(os.listdir(l_dir))[::2]
    r_f = sorted(os.listdir(r_dir))[::2]
    d_f = sorted(os.listdir(d_dir))
    
    while True:
        for i in range(int(0.8*len(l_f))):
            d = np.clip(np.expand_dims(np.expand_dims(cv2.imread(d_dir+d_f[i],0).astype('float32'),axis=0),axis=0),0,max_disp)
            l = np.expand_dims(np.expand_dims(cv2.imread(l_dir+l_f[i],0).astype('float32')/255,axis=0),axis=0)
            r0 = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
            
            ret = (np.concatenate((l,r0),axis=1),d)
            yield ret
            #del ret

def generate_validation_data(train_dir = '/home/nownow/Documents/projects/stereo/data/training/'):
    
    max_disp = 256
    
    #w = int(np.random.normal(600,60))
    #h = int(np.random.normal(150,30))
    
    l_dir = train_dir+'image_2/'
    r_dir = train_dir+'image_3/'
    d_dir = train_dir+'disp_noc_0/'
    
    l_f = sorted(os.listdir(l_dir))[::2]
    r_f = sorted(os.listdir(r_dir))[::2]
    d_f = sorted(os.listdir(d_dir))
    
    while True:
        for i in range(int(0.8*len(l_f))):
            d = np.clip(np.expand_dims(np.expand_dims(cv2.imread(d_dir+d_f[i],0).astype('float32'),axis=0),axis=0),0,max_disp)
            l = np.expand_dims(np.expand_dims(cv2.imread(l_dir+l_f[i],0).astype('float32')/255,axis=0),axis=0)
            r0 = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
            
            ret = (np.concatenate((l,r0),axis=1),d)
            yield ret
            
#d = []
#l = []
#r = []
#
#shfl = 1 + np.random.permutation(25000)
#
#for i in range(50):
#    d.append(cv2.imread("/home/nownow/stereo_data/Depth_map/DepthMap_"+str(shfl[i])+".png",0).astype('float32'))
#    l.append(cv2.imread("/home/nownow/stereo_data/StereoImages/Stereoscopic_"+str(shfl[i])+"_L.png",0).astype('float32')/255)
#    r.append(cv2.imread("/home/nownow/stereo_data/StereoImages/Stereoscopic_"+str(shfl[i])+"_R.png",0).astype('float32')/255)
#    
#print "Dataset loaded"
#
#d = np.clip(np.expand_dims(np.array(d),axis=1),0,10)/10.0
#im = np.append(np.expand_dims(np.array(l),axis=1),np.expand_dims(np.array(r),axis=1),axis=1)

x = Input(shape=(2,None,None))
conv1 = Conv2D(16,(3,3),padding="same",activation='relu')(x)
pool1 = MaxPooling2D()(conv1)
conv2 = Conv2D(32,(3,3),padding="same",activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2)
conv3 = Conv2D(64,(3,3),padding="same",activation='relu')(pool2)
pool3 = MaxPooling2D()(conv3)
conv4 = Conv2D(128,(3,3),padding="same",activation='relu')(pool3)
conv5 = Conv2D(32,(3,3),padding="same",activation='relu')(conv4)
up5 = UpSampling2D()(conv5)
conv6 = Conv2D(16,(3,3),padding="same",activation='relu')(up5)
up6 = UpSampling2D()(conv6)
conv7 = Conv2D(8,(3,3),padding="same",activation='relu')(up6)
up7 = UpSampling2D()(conv7)
y = Conv2D(1,(3,3),padding="same",activation='relu')(up7)

model = Model(x,y)

model.compile(optimizer='adadelta',loss='mean_absolute_error')

#model.fit(im,d,validation_split=0.2,epochs = 10,batch_size=10)
model.fit_generator(generator=generate_training_data(),
                    steps_per_epoch=160,
                    epochs=20,
                    #callbacks=callback,
                    validation_data=generate_validation_data(),
                    validation_steps=40,
                    max_queue_size=1)