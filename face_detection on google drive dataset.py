# -*- coding: utf-8 -*-
"""Face Detection on dataset loaded on google drive

"""

from google.colab import drive
drive.mount('/content/gdrive')

from google.colab import drive
drive.mount('/content/drive/Colab Notebooks/ dataset')

! pip install Augmentor
import torchvision.models as models
from torch import nn
import torch
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import Augmentor
import os
import cv2
from sklearn.utils import shuffle

def label(inp):
    return int(inp.split('/')[-1].split('_')[0])

class dataset():
    def __init__(self, data_path , kind , transform = None):
        
        self.trainimages = glob.glob(data_path + 'train/*.png')
        self.valimages = glob.glob(data_path + 'val/*.png')
        self.testimages = glob.glob(data_path + 'test/*.png')
        self.transform = transform
        
        if kind =='train':
            self.data = self.trainimages
        elif kind == 'validation':
            self.data = self.valimages
        elif kind == 'test':
            self.data = self.testimages
        else:
            raise Exception('Kind is not valid !')
        
        
    def __getitem__(self,index):

        im = Image.open(self.data[index])
        targ = torch.LongTensor([label(self.data[index])])

        if self.transform:
            im = self.transform(im)
            
#         im = torch.FloatTensor(np.array(im))     
        im = transforms.ToTensor()(np.array(im)).float()  
        return im ,targ
    
    
    def __len__(self):
        return len(self.data)

    
    
piplin = Augmentor.Pipeline()
piplin.crop_random(0.5,0.8)
piplin.flip_left_right(0.3)
piplin.histogram_equalisation(1)
piplin.random_contrast(0.7, 0.2, 1.2)
piplin.invert(0.3)
piplin.random_brightness(0.7, 0.2, 1.2)
piplin.random_erasing(0.8, 0.5)
piplin.rotate(0.6, 15, 15)
piplin.shear(0.5, 5, 5)
piplin.skew(0.6, magnitude=0.1)
piplin.resize(1, 224, 224, resample_filter=u'ANTIALIAS')
    
    
T = transforms.Compose([
    piplin.torch_transform(),
])


data_path = './gdrive/My Drive/Dataset/'
train_dataset = dataset(data_path,kind='train',transform = T)
val_dataset = dataset(data_path,kind='validation')
test_dataset = dataset(data_path,kind='test')

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = 4,shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,batch_size = 2,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = 2,shuffle = False)


resnet = models.resnet34(pretrained=False)
resnet = resnet.train()
resnet.fc = nn.Linear(512,256)
resnet = nn.Sequential(resnet, nn.Linear(256,128))
resnet = nn.Sequential(resnet, nn.Linear(128,64))
resnet = nn.Sequential(resnet, nn.Linear(64,32))
resnet = nn.Sequential(resnet, nn.Linear(32,20))
resnet = nn.Sequential(resnet, nn.Linear(20,10))
Model = nn.Sequential(resnet, nn.Linear(10,6))

CUDA = torch.cuda.is_available()

if CUDA:
    Model = Model.cuda()
    
Loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(),lr = 0.0001)

epoch = 2500
desiredecc = 75
iter = 0
for i in range(epoch):
    for Images,targs in train_loader:
#         print(targs.squeeze(1))
        iter+=1
        if CUDA:
            Images = Variable(Images.cuda())
            targs = Variable(targs.cuda())
        else:
            Images = Variable(Images)
            targs = Variable(targs)
            
        optimizer.zero_grad()
        outputs = Model(Images)
        loss = Loss_fn(outputs , targs.squeeze(1))
        loss.backward()
        optimizer.step()
        
        
        
        if (iter+1)% 150 == 0:
            correct = 0
            total = 0
            for images,labels in val_loader:
                if CUDA:
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                outputs = Model(images)
                _,predicted = torch.max(outputs.data,1)
                labels = labels.squeeze(1)
                total += labels.size(0)
                if CUDA:
                    correct += (predicted.cpu()==labels.cpu()).sum()
                else:
                    correct += (predicted==labels).sum()
                    
            accuracy = 100 * correct / total
            print('at epoch {} accuracy is {} %'.format(i,accuracy))   
            if accuracy >= desiredecc:
              optimizer = torch.optim.SGD(Model.parameters(), lr = 0.00001)
            if accuracy > 80:
                torch.save(Model.state_dict(),'./gdrive/My Drive/Model.pt')

torch.save(Model.state_dict(),'./gdrive/My Drive/Model.pt')

