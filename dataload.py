# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:42:38 2021

@author: BanChunggi
"""

# PyTorch에서 제공하는 ImageFolder (library)
# 각 이미지들이 자신의 레이블(label) 이름으로 된 폴더 안에 들어가 있는 구조라면, 
# 이를 바로 불러와 객체로 만들 수 있다.

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

'''
img = Image.open('data/train/banana/Banana01.png')
plt.imshow(img)
plt.show()
#data 시각화
'''

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)) 
    ])

train_dataset = datasets.ImageFolder(root='data/train/', transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

classes = train_dataset.classes

print(classes)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(inputs[0].shape)    
    custom_imshow(inputs[0])
    #inputs[0]은 inputs의 이미지 하나를 의미 (3, 224, 224)
    pass
