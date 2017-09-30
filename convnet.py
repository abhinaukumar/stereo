# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:37:54 2017

@author: nownow
"""

import keras
#import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, UpSampling2D, Flatten, Merge, Reshape, BatchNormalization, Lambda
from keras.models import Model, Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

plt.ion()
max_disp = 256

#def generate_train_data(train_dir = '/home/nownow/Documents/projects/stereo/data/training/'):
#    w = int(np.random.normal(600,60))
#    h = int(np.random.normal(150,30))
#    
#    l_dir = train_dir+'image_2/'
#    r_dir = train_dir+'image_3/'
#    d_dir = train_dir+'disp_noc_0/'
#    
#    l_f = sorted(os.listdir(l_dir))[::2]
#    r_f = sorted(os.listdir(r_dir))[::2]
#    d_f = sorted(os.listdir(d_dir))
#    
#    while True:
#        for i in range(int(0.8*len(l_f))):
#            d = np.clip(np.expand_dims(np.expand_dims(cv2.imread(d_dir+d_f[i],0).astype('float32'),axis=0),axis=0),0,max_disp)
#            l = np.expand_dims(np.expand_dims(cv2.imread(l_dir+l_f[i],0).astype('float32')/255,axis=0),axis=0)
#            r0 = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
#            
#            d = d[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
#            l = l[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
#            r0 = r0[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
#            
#            x = [l,r0]
#            for i in range(max_disp-1):
#                temp = np.zeros(r0.shape)
#                temp[:,:,:,1+i:] = r0[:,:,:,:-(i+1)]
#                x.append(temp)
#            ret = (x,d)
#            yield ret
#            del ret
    
def generate_training_data(train_dir = '/home/nownow/Documents/projects/stereo/data/training/'):
    #w = int(np.random.normal(600,60))
    #h = int(np.random.normal(150,30))

    w = 32
    h = 32   
    
    cx = int(np.random.uniform(400,1200-w/2))
    cy = int(np.random.uniform(150,300-h/2))

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
            r = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
                
            d = d[:,:,cy-h/2:cy+h/2,cx-w/2:cx+w/2]
            l = l[:,:,cy-h/2:cy+h/2,cx-w/2:cx+w/2]
            #r0 = r[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
            
            x = np.zeros((1,2,max_disp,d.shape[-2],d.shape[-1]))
            for j in range(max_disp):
                x[0,0,j,:,:] = l
                
            #x = [l,r0]
            for j in range(max_disp):
                #temp = np.zeros(r0.shape)
                #x[0,1,i,:,1+i:] = r0[:,:,:,:-(i+1)]
                x[0,1,j,:,:] = r[:,:,cy-h/2:cy+h/2,cx-w/2-j:cx+w/2-j]
                #x.append(temp)
            d = keras.utils.to_categorical(d,max_disp)
            d = np.reshape(d,[1,max_disp] + list(l[0][0].shape))
            ret = (x,d)
            yield ret
            #del ret

def generate_validation_data(train_dir =  '/home/nownow/Documents/projects/stereo/data/training/'):
    #w = int(np.random.normal(600,60))
    #h = int(np.random.normal(150,30))
    
    w = 32
    h = 32

    cx = int(np.random.uniform(400,1200-w/2))
    cy = int(np.random.uniform(150,300-h/2))

    l_dir = train_dir+'image_2/'
    r_dir = train_dir+'image_3/'
    d_dir = train_dir+'disp_noc_0/'
    
    l_f = sorted(os.listdir(l_dir))[::2]
    r_f = sorted(os.listdir(r_dir))[::2]
    d_f = sorted(os.listdir(d_dir))
    
    while True:
        for i in range(int(0.8*len(l_f)),len(l_f)):
            d = np.clip(np.expand_dims(np.expand_dims(cv2.imread(d_dir+d_f[i],0).astype('float32'),axis=0),axis=0),0,max_disp)
            l = np.expand_dims(np.expand_dims(cv2.imread(l_dir+l_f[i],0).astype('float32')/255,axis=0),axis=0)
            r = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
                
            d = d[:,:,cy-h/2:cy+h/2,cx-w/2:cx+w/2]
            l = l[:,:,cy-h/2:cy+h/2,cx-w/2:cx+w/2]
            #r0 = r[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
            
            x = np.zeros((1,2,max_disp,d.shape[-2],d.shape[-1]))
            for j in range(max_disp):
                x[0,0,j,:,:] = l
                
            #x = [l,r0]
            for j in range(max_disp):
                #temp = np.zeros(r0.shape)
                #x[0,1,i,:,1+i:] = r0[:,:,:,:-(i+1)]
                x[0,1,j,:,:] = r[:,:,cy-h/2:cy+h/2,cx-w/2-j:cx+w/2-j]
                #x.append(temp)
            d = keras.utils.to_categorical(d,max_disp)
            d = np.reshape(d,[1,max_disp] + list(l[0][0].shape))
            ret = (x,d)
            yield ret
            #del ret            

#def generate_validation_data(train_dir = '/home/nownow/Documents/projects/stereo/data/training/'):
#    w = int(np.random.normal(600,60))
#    h = int(np.random.normal(150,30))
#    
#    l_dir = train_dir+'image_2/'
#    r_dir = train_dir+'image_3/'
#    d_dir = train_dir+'disp_noc_0/'
#    
#    l_f = sorted(os.listdir(l_dir))[::2]
#    r_f = sorted(os.listdir(r_dir))[::2]
#    d_f = sorted(os.listdir(d_dir))
#    
#    while True:
#        for i in range(int(0.8*len(l_f)),len(l_f)):
#            d = np.clip(np.expand_dims(np.expand_dims(cv2.imread(d_dir+d_f[i],0).astype('float32'),axis=0),axis=0),0,max_disp)
#            l = np.expand_dims(np.expand_dims(cv2.imread(l_dir+l_f[i],0).astype('float32')/255,axis=0),axis=0)
#            r0 = np.expand_dims(np.expand_dims(cv2.imread(r_dir+r_f[i],0).astype('float32')/255,axis=0),axis=0)
#            
#            d = d[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
#            l = l[:,:,150-h/2:150+h/2,600-w/2:600+w/2]
#            r0 = r0[:,:,150-h/2:150+h/2,600-w/2:600+w/2]            
#            
#            x = [l,r0]
#            for i in range(max_disp-1):
#                temp = np.zeros(r0.shape)
#                temp[:,:,:,1+i:] = r0[:,:,:,:-(i+1)]
#                x.append(temp)
#            yield (x,d)

#max_disp = 10
#
#d = []
#l = []
#r0 = []
#
#l_dir = '/home/nownow/Documents/projects/stereo/data/training/image_2/'
#r_dir = '/home/nownow/Documents/projects/stereo/data/training/image_3/'
#d_dir = '/home/nownow/Documents/projects/stereo/data/training/disp_noc_0/'
#
#l_f = sorted(os.listdir(l_dir))
#r_f = sorted(os.listdir(r_dir))
#d_f = sorted(os.listdir(d_dir))
#
#l_f = (l_f[::2])[:155]
#r_f = (r_f[::2])[:155]
#d_f = d_f[:155]

#shfl = 1 + np.random.permutation(25000)

#for i in range(5):
#    d.append(cv2.imread("/home/nownow/stereo_data/Depth_map/DepthMap_"+str(shfl[    i])+".png",0).astype('float32')/255)
#    l.append(cv2.imread("/home/nownow/stereo_data/StereoImages/Stereoscopic_"+str(shfl[i])+"_L.png",0).astype('float32')/255)
#    r0.append(cv2.imread("/home/nownow/stereo_data/StereoImages/Stereoscopic_"+str(shfl[i])+"_R.png",0).astype('float32')/255)

#for i in range(len(l_f)):
#    d.append(cv2.imread(d_dir+d_f[i],0).astype('float32'))
#    l.append(cv2.imread(l_dir+l_f[i],0).astype('float32')/255)
#    r0.append(cv2.imread(r_dir+r_f[i],0).astype('float32')/255)
#
#print "Dataset loaded"
#
#d = np.clip(np.expand_dims(np.array(d),axis=1),0,10)/10.0
#l = np.expand_dims(np.array(l),axis=1)
#r0 = np.expand_dims(np.array(r0),axis=1)
#
#print r0.shape
#
#r = [r0]
#
#for i in range(255):
#    temp = np.zeros(r0.shape)
#    temp[:,:,:,1+i:] = r0[:,:,:,:-(i+1)]
#    r.append(temp)
#    
#
#x = [l]
#x.extend(r)

def argmin(cost_mat):
    return keras.backend.expand_dims(keras.backend.cast(keras.backend.argmin(cost_mat,axis = 1),dtype='float32')/10,axis=1)

def argmin_output_shape(shapes):
    return (shapes[0],1,shapes[2],shapes[3])

def expand1(tensor):
    return keras.backend.expand_dims(tensor,axis=1)

def expand2(tensor):
    return keras.backend.expand_dims(tensor,axis=2)

def contract(tensor):
    return tensor[:,0,:,:,:]
    
seq = Sequential()

seq.add(Conv3D(4,(1,7,7),padding="same",activation='relu',input_shape=(2,None,None,None)))
#seq.add(MaxPooling2D())
seq.add(Conv3D(8,(1,5,5),padding="same",activation='relu'))
#seq.add(MaxPooling2D())
seq.add(Conv3D(16,(1,3,3),padding="same",activation='relu'))
#seq.add(MaxPooling2D())
seq.add(Conv3D(16,(1,3,3),padding="same",activation='relu'))
#seq.add(Conv2D(4,(7,7),padding="same",activation='relu'))
#seq.add(UpSampling2D())
seq.add(Conv3D(8,(1,5,5),padding="same",activation='relu'))
#seq.add(UpSampling2D())
seq.add(Conv3D(4,(1,7,7),padding="same",activation='relu'))
#seq.add(UpSampling2D())
#seq.add(Conv2D(1,(3,3),padding="same",activation='relu'))

seq.add(Conv3D(1,(1,9,9),padding="same",activation='relu'))

#seq.trainable = False

#left_input = Input(shape=(1,None,None))
#right_inputs = []

#for i in range(max_disp):
#    right_inputs.append(Input(shape=(1,None,None)))

#_costs = []

#_x = []
#for i in range(max_disp):
#    _costs.append(seq(keras.layers.concatenate([left_input,right_inputs[i]],axis=1)))

#for i in range(max_disp):
#    _x.append(Lambda(expand2,output_shape=(2,1,None,None))(keras.layers.concatenate([left_input,right_inputs[i]],axis=1)))

#x = keras.layers.concatenate(_x,axis=2)
x = Input(shape=(2,None,None,None))
costs = seq(x)

#costs = keras.layers.concatenate(_costs,axis=1)
#costs = Lambda(expand,output_shape=(1,256,None,None))(costs)

#smoothed_costs = Conv2D(max_disp,(7,7),padding="same",activation='relu')(costs)
    
#smoothed_costs = Lambda(expand,output_shape=(1,256,None,None))(smoothed_costs)
#smoothed_costs = Conv2D(max_disp,(5,5),padding="same",activation='relu')(costs)
smoothed_costs = Conv3D(4,(5,5,5),padding="same",activation='relu')(costs)
smoothed_costs = Conv3D(8,(3,3,3),padding="same",activation='relu')(smoothed_costs)
smoothed_costs = Conv3D(4,(3,3,3),padding="same",activation='relu')(smoothed_costs)
smoothed_costs = Conv3D(1,(5,5,5),padding="same",activation='sigmoid')(smoothed_costs)
#smoothed_costs = Lambda(contract,output_shape=(256,None,None))(smoothed_costs)
y = Lambda(contract,output_shape=(max_disp,None,None))(smoothed_costs)
#y = Conv2D(1,(3,3),activation='relu',padding='same')(keras.backend.reshape(smoothed_costs,(max_disp,None,None)))
#y = Conv2D(1,(3,3),activation='relu',padding='same')(smoothed_costs)
#y = Lambda(argmin,output_shape=argmin_output_shape)(costs)
#y = costs

#inputs = [left_input]
#inputs.extend(right_inputs)

#model = Model(inputs,y)
model = Model(x,y)

model.summary()

model.compile(optimizer='adadelta',loss='binary_crossentropy')
callback = [keras.callbacks.ModelCheckpoint('/home/nownow/Documents/projects/stereo/saved_files/sep_30_16_53.h5',save_best_only=True,save_weights_only=True),
	    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=3, min_lr=0.001)]

#model.fit_generator(,d,epochs=15,callbacks=callback)
model.fit_generator(generator=generate_training_data(),
                    steps_per_epoch=160,
                    epochs=100,
                    callbacks=callback,
                    validation_data=generate_validation_data(),
                    validation_steps=40)
                    #max_queue_size=1)
