# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:42:38 2021

@author: BanChunggi
"""

# PyTorch에서 제공하는 ImageFolder (library)
# 각 이미지들이 자신의 레이블(label) 이름으로 된 폴더 안에 들어가 있는 구조라면, 
# 이를 바로 불러와 객체로 만들 수 있다.

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

img = Image.open('data/train/banana/Banana01.png')
plt.imshow(img)
plt.show()

#data 시각화

