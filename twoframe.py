#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:19:00 2021

@author: songming
"""

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# params for ShiTomasi corner detection
feature_paramd = dict( maxCorners = 10,
                       qualityLevel = 0.1,
                       minDistance = 3,
                       )

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.1,
                       minDistance = 3,
                       )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

res = []
def computeFundamentalMatrix(kps_ref, kps_cur):             
    F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, 2, 0.98)
    if F is None or F.shape == (1, 1):
        # no fundamental matrix found
        raise Exception('No fundamental matrix found')
    elif F.shape[0] > 3:
        # more than one matrix found, just pick the first
        print('more than one matrix found, just pick the first')
        F = F[0:3, 0:3]
    return np.matrix(F), mask 

box50 = np.load('03box.npy')
score50 = np.load('03score.npy')
classes50 = np.load('03class.npy')
valid50 = np.load('03valid_detections.npy')


box55 = np.load('04box.npy')
score55 = np.load('04score.npy')
classes55 = np.load('04class.npy')
valid55 = np.load('04valid_detections.npy')

img1 = cv2.imread('0000000003.png')
img2 = cv2.imread('0000000004.png')

image_h, image_w, _ = img1.shape

# gloabl mask for static parts of the scene
masks = np.ones(img1.shape[:2],np.uint8)

# global list container for corners in movable objects
# save the object's index in the list during semantic RANSAC
# instance-level object is saved as an array in the list
# for each instance, it is composed by corner points in the numpy array
movable_list1 = []
movable_list2 = []

# global list container for corners in static objects
static_list1 = []
static_list2 = []

# global list container for corners in the background
background_list1 = []
background_list2 = []

#pay attention to the gloabl and local variable with the same name
for i in range(valid50[0]):        
    coor = np.copy(box50[0][i])    
    coor[0] = int(coor[0] * image_h)
    coor[2] = int(coor[2] * image_h)
    coor[1] = int(coor[1] * image_w)
    coor[3] = int(coor[3] * image_w)
    masks[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] = 0
    if score50[0][i] < 0.4: # threshold to filter out low probability bboxes
        continue
    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])        
    cv2.rectangle(img1, c1, c2, (255,0,0), 1)
    maskd = np.zeros(img1.shape[:2],np.uint8)
    maskd[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] = 1
    
    corners_1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), mask = maskd, **feature_paramd)
    corners_2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                       cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners_1, None, **lk_params)
    corners_1 = corners_1[st==1]
    movable_list1.append(corners_1)
    corners_2 = corners_2[st==1]
    movable_list2.append(corners_2)
    
    # for corner in corners_1:
    #     x,y = corner.ravel()
    #     cv2.circle(img1,(x,y),3,(0,0,255),-1)    

    # for corner2 in corners_2:
    #     x2,y2 = corner2.ravel()
    #     cv2.circle(img2,(x2,y2),3,(0,0,255),-1)   
        
for i in range(valid55[0]):
    if score55[0][i] < 0.4:
        continue
    coor = np.copy(box55[0][i])
    coor[0] = int(coor[0] * image_h)
    coor[2] = int(coor[2] * image_h)
    coor[1] = int(coor[1] * image_w)
    coor[3] = int(coor[3] * image_w)
    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])        
    cv2.rectangle(img2, c1, c2, (255,0,0), 1)
    
    
#exclude the bounding box area to find static background points for F matrix estimation       
if (1):   #use local parameters, no conflict with others
    corners_1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), mask = masks, **feature_params)
    corners_2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                       cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners_1, None, **lk_params)
    corners_1 = corners_1[st==1]
    background_list1.append(corners_1)
    corners_2 = corners_2[st==1]
    background_list2.append(corners_2)    

    # for corner in corners_1:
    #     x,y = corner.ravel()
    #     cv2.circle(img1,(x,y),3,(255,0,0),-1)    # blue corner points drawing in prev frame

    # for corner2 in corners_2:
    #     x2,y2 = corner2.ravel()
    #     cv2.circle(img2,(x2,y2),3,(255,0,0),-1)  # blue corner points drawing in curr frame
        
    F,mask = computeFundamentalMatrix(corners_1, corners_2)
    
    for obj_corners1, obj_corners2 in zip(movable_list1, movable_list2):
        corners1 = np.concatenate((obj_corners1, corners_1), axis=0)
        corners2 = np.concatenate((obj_corners2, corners_2), axis=0)
        F_esti, mask_movable = computeFundamentalMatrix(corners1, corners2)
        a = 1-sum(mask_movable[0:len(obj_corners1)])/len(obj_corners1)
        res.append(a)
        if a > 0.5:
            for tmp in obj_corners1:
                x,y = tmp.ravel()
                cv2.circle(img2,(x,y),3,(0,0,255),-1)
        # Epipolar equation x'Fx=0
        
        
        
    
#concatenate the corners points from background parts and chosen bounding box, evaluate the F matrix and 
#see the mask of inlier(1) and outlier(0), save the number of background points, the rest are on the movable
#objects 
   
#find corner points in the bounding boxes and track these points to next frame

#1. Define the ROI img1[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] according to the bbox coordinantes

#2. Find good features to tarck with KLT in the ROI

#3. Calculate the opipolar geometry redisudals with the established correspondence points

#4. Decision-making for moving objects based on the outliers proportion in the boundong box


# t = time.time()
    #run a backward-check for match verification between frames
    # p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    # p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    # d = abs(p0-p0r).reshape(-1, 2).max(-1)
    # status = d < back_threshold
# computingtime = time.time() - t

img1 = cv2.resize(img1, (0, 0), None, 1, 1)
img2 = cv2.resize(img2, (0, 0), None, 1, 1)
numpy_horizontal = np.vstack((img1, img2))
#numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)    
# cv2.imshow('KLT tracking',numpy_horizontal)
cv2.imwrite('/home/songming/kitti.png', numpy_horizontal)
# cv2.waitKey()
# cv2.destroyAllWindows()
