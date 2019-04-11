from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.ndimage.map_coordinates

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
    picture = torch.from_numpy(first_image).float()/255
    optical_flow = torch.from_numpy(flow)
    pdb.set_trace()
    F.grid_sample(picture, optical_flow)
    pdb.set_trace()
    good_new = p1[st==1]
    good_old = p0[st==1]
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # cv2.imshow('frame',mask)
