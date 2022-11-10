#!/usr/bin/env python
# coding: utf-8

# In[49]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image


# In[50]:


def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()


# In[51]:


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


# In[52]:


def equalHist(img):
    # 灰度图像矩阵的高、宽
    h,w,x = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


# In[53]:


bacepath = "D:/html_code/NTU/COVID-19_Radiography_Dataset/Normal"
savepath = 'D:/html_code/NTU/COVID-19_Radiography_Dataset/Normal1'


# In[54]:


path_list = os.listdir(bacepath)


# In[55]:


for filename in path_list:
    print(filename)
    img = cv.imread(bacepath + '/' + filename)
    equa = equalHist(img)
    cv.imwrite(savepath + "/" + filename, equa)

