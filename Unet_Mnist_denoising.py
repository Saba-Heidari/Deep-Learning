# -*- coding: utf-8 -*-
"""Unet Saba:
Use UNet autoencoder in denoising images.
In this code,  80% Gaussian noise added to the MNIST dataset in the training dataset.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Dense, LeakyReLU, UpSampling2D, Input, Dropout, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy import squeeze, expand_dims

(x_train,y_train), (x_test, y_test) = mnist.load_data()

# x_ = expand_dims(x_train[0], axis=2)
x_train = expand_dims(x_train, axis = 3)
# x_test = expand_dims(x_test, axis = 3)



x_test[0].shape

def zpad(x):
  return expand_dims(np.pad(x.squeeze(), ((2,2),(2,2)), 'constant'),axis = 2)

x_train_new = np.zeros((x_train.shape[0],32,32,1))
x_test_new = np.zeros((x_test.shape[0],32,32,1))
for i in range(len(x_train)):
  x_train_new[i] = zpad(x_train[i])
for i in range(len(x_test)):
  x_test_new[i] = zpad(x_test[i])

plt.imshow(x_train_new[0].squeeze())

def partial_noise(image,sigma, noise_proportion):
  length = int(image.flatten().shape[0])
  len_noise = int(noise_proportion*length)
  
  address = list(np.random.choice(int(length), int(len_noise),replace=False))
  image = image.flatten()
  noise = sigma*np.random.randn(len_noise)
  image[[address]] = image[[address]] + noise
  image = expand_dims(image.reshape(-1,32), axis= 3)
  return image


# add noise
sigma = 40
noise_proportion=0.8
x_train_noisy = np.zeros((x_train_new.shape[0],x_train_new.shape[1],x_train_new.shape[2],1))
x_test_noisy = np.zeros((x_test_new.shape[0],x_test_new.shape[1],x_test_new.shape[2],1))

for i in range(len(x_train_new)):
  x_train_noisy[i] = partial_noise(x_train_new[i],sigma = sigma, noise_proportion=noise_proportion)
print('Training set contaminated...!')
for i in range(len(x_test_new)):
  x_test_noisy[i] = partial_noise(x_test_new[i],sigma = sigma, noise_proportion=noise_proportion)
print('Test set contaminated...!')

plt.imshow(x_train_noisy[0].squeeze())

def unet(pretrained_weights = None,input_size = (32,32,1)):
   
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    

    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
  

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    
    #pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2,2), padding='same')(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    #pool5 = MaxPooling2D(pool_size=(2,2),padding='same')(drop5)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    #up7 = Conv2DTranspose(256, 3, strides=(1, 1), padding='same',)(UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([conv5,conv6], axis = 3)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    #up8 = Conv2DTranspose(256, 3, strides=(1, 1), padding='same',)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv4,conv7], axis = 3)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)


    #up9 = Conv2DTranspose(128, 3, strides=(1, 1), padding='same',)(UpSampling2D(size = (2,2))(conv8))
   
    merge9 = concatenate([conv3,conv8], axis = 3)

    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    up10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',)(conv9)
    merge10 = concatenate([conv2,up10], axis = 3)

    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    up11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',)(conv10)
    merge11 = concatenate([conv1,up11], axis = 3)


    conv12 = Conv2D(1, 1, activation = 'sigmoid')(merge11)
    # dens = Dense(units=10, activation = 'softmax')(conv12) 
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'],)

    return  model

model = unet(input_size=(32,32,1))
# print(unet(input_size=(28,28,1)))



    
print(model.summary())
   


epochs = 3
batch_size = 200


model = unet(input_size = (32,32,1))
model.fit(x = x_train_noisy, y = x_train_new , batch_size= batch_size  ,epochs = epochs, validation_data = (x_test_noisy, x_test_new ), verbose=1)

output_1 =  model.predict(x_test_noisy,batch_size= 20)[0]
plt.imshow(output_1.squeeze())

plt.imshow(x_train_noisy[0].squeeze())

epochs = 10
batch_size = 2000


model2 = unet_2(input_size = (32,32,1))
model2.summary()
