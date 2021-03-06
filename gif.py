from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import datetime

import zipfile
import time
import copy
import os
import cv2
import numpy as np

def show(image,title=None):
    plt.figure()
    imshow(image, title=title)


if __name__ == "__main__":
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    first_image = cv2.imread("./jump_original/frame_00_delay-0.08s.jpg")
    second_image = cv2.imread("./jump_original/frame_01_delay-0.08s.jpg")
    color = np.random.randint(0, 255, (100, 3))
    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, p0, None, **lk_params)
    flow = cv2.calcOpticalFlowFarneback(first_gray,second_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    pdb.set_trace()
    magnitude = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    threshed = cv2.threshold(magnitude,127,255,cv2.THRESH_BINARY)
    show(threshed)
    cv2.imshow('frame2', bgr)
    k = cv2.waitKey(30) & 0xff
    pdb.set_trace()
    hsv = np.zeros_like(second_image)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)
    k = cv2.waitKey(30) & 0xff
    pdb.set_trace()
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # cv2.imshow('frame',mask)
