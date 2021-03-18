#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:19:00 2021

@author: haixin
"""

import cv2
import numpy as np
import time
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def computeFundamentalMatrix(kps_ref, kps_cur):
             
    F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, 2, 0.99)
    if F is None or F.shape == (1, 1):
        # no fundamental matrix found
        raise Exception('No fundamental matrix found')
    elif F.shape[0] > 3:
        # more than one matrix found, just pick the first
        print('more than one matrix found, just pick the first')
        F = F[0:3, 0:3]
    return np.matrix(F), mask 

box50 = np.load('box50.npy')
score50 = np.load('score50.npy')
classes50 = np.load('class50.npy')
valid50 = np.load('valid_detections50.npy')

img1 = cv2.imread('n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605012404.jpg')

img2 = cv2.imread('n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg')

image_h, image_w, _ = img1.shape
for i in range(valid50[0]):
    coor = box50[0][i]
    coor[0] = int(coor[0] * image_h)
    coor[2] = int(coor[2] * image_h)
    coor[1] = int(coor[1] * image_w)
    coor[3] = int(coor[3] * image_w)
    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
    cv2.rectangle(img1, c1, c2, (0,0,255), 1)
    
#find corner points in the bounding boxes and track these points to next frame
#1. Define the ROI img1[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] 
#   according to the bbox coordinantes

#2. Find good features to tarck in the ROI

t = time.time()
corners_1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),150,0.01,10) # maxCorners, qualityLevel, minDistance
corners_2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                       cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners_1, None, **lk_params)

corners_1 = corners_1[st==1]
corners_2 = corners_2[st==1]

    
F,mask = computeFundamentalMatrix(corners_1,corners_2)

computingtime = time.time() - t
#cv2.circle(img1,c1,3,(0,0,255),4)
#cv2.circle(img1,c2,3,(0,0,255),4)
for corner in corners_1:
    x,y = corner.ravel()
    cv2.circle(img1,(x,y),3,255,2)
cv2.imshow('corner',img1)

# for corner in corners_2:
#     x,y = corner.ravel()
#     cv2.circle(img2,(x,y),3,255,-1)
# cv2.imshow('corner',img2)
cv2.waitKey()
cv2.destroyAllWindows()
