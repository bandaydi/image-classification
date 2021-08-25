# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:29:49 2021

@author: BanChunggi
"""

import numpy as np
import matplotlib.pyplot as plt

def custom_imshow(img):
    print(img.shape)
    img = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()