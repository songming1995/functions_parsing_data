#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:19:00 2021

@author: haixin
"""

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from detector import detector
import os
import core.utils as utils
from PIL import Image
#params for ShiTomasi corner detection
feature_paramd = dict( maxCorners = 15,
                        qualityLevel = 0.2,
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

# image_list = os.listdir('/home/songming/tensorflow-yolov4-tflite/Sequence_071/images')
image_list = os.listdir('/home/songming/tensorflow-yolov4-tflite/Sequence_05/images')
image_list.sort()


# write the images to a video
# img_array = []

class_file_name = '/home/songming/tensorflow-yolov4-tflite/data/classes/coco.names'
names = {}  # dictionary to save the class label
with open(class_file_name, 'r') as data:
    for ID, name in enumerate(data):
        names[ID] = name.strip('\n')

num_classes = len(names)
static_object = [9, 10, 11, 12, 13]


for k in range(0,len(image_list)-1):
# for k in range(0,2):
    
    file_name1, ext1 = os.path.splitext(image_list[k]) #split filename and format
    file_name2, ext2 = os.path.splitext(image_list[k+1]) #split filename and format    
    
    img1 = cv2.imread(os.path.join('/home/songming/tensorflow-yolov4-tflite/Sequence_05/images', image_list[k]))
    img2 = cv2.imread(os.path.join('/home/songming/tensorflow-yolov4-tflite/Sequence_05/images', image_list[k+1]))
    image_h, image_w, _ = img1.shape 
    #convert to grayscale
    img1_gr = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gr = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # adaptive histogram equalization to improve the contrast of our images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1_greq = clahe.apply(img1_gr)
    img2_greq = clahe.apply(img2_gr)
  
    
    box1 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name1 + 'box.npy')    
    score1 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name1  + 'score.npy')        
    classes1 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name1 + 'classes.npy')    
    valid1 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name1 + 'valid.npy')

    box2 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name2 + 'box.npy')    
    score2 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name2  + 'score.npy')        
    classes2 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name2 + 'classes.npy')    
    valid2 = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name2 + 'valid.npy')
    
    # mask for background parts of the scene
    maskb2 = np.ones(img2.shape[:2],np.uint8)  #background mask, set all ones initially
    
    
    # global list container for corners in movable objects
    # save the object's index in the list during semantic RANSAC
    
    # instance-level object is saved as an array in the list
    # for each instance, it is composed by corner points in the numpy array
    
    # global list container for corners in movable objects    
    movable_list1 = []
    movable_list2 = []
    
    # global list container for coordinates of movable objects 
    movable_list1_coor = []
    movable_list2_coor = []
    
    
    # global list container for corners in static objects
    static_list1 = []
    static_list2 = []
    
    # global list container for coordinates of static objects
    static_list1_coor = []
    static_list2_coor = []
    
    # global list container for corners in the background
    # background_list1 = []
    # background_list2 = []
    
    #pay attention to the gloabl and local variable with the same name

        #to do
        #put coor in a list -> static_list2_coor and movable_list2_coor 
        #put class name in a list for semantic check, outlier rejection
        
       
        
    for i in range(valid2[0]):
        if score2[0][i] < 0.2: # threshold to filter out low probability bboxes
            continue        
        coor2 = np.copy(box2[0][i])    
        coor2[0] = int(coor2[0] * image_h)
        coor2[2] = int(coor2[2] * image_h)
        coor2[1] = int(coor2[1] * image_w)
        coor2[3] = int(coor2[3] * image_w) 
        
        class_ind2 = int(classes2[0][i])

        
        
        if class_ind2 in static_object:
            static_list2_coor.append(coor2)
            masks2 = np.zeros(img2.shape[:2],np.uint8)
            masks2[int(coor2[0]):int(coor2[2]), int(coor2[1]):int(coor2[3])] = 1
            corners_2 = cv2.goodFeaturesToTrack(img2_greq, mask = masks2, **feature_paramd)
            corners_1, st, err = cv2.calcOpticalFlowPyrLK(img2_greq, img1_greq, corners_2, None, **lk_params)
            corners_2 = corners_2[st==1]
            static_list2.append(corners_2)       
            corners_1 = corners_1[st==1]
            static_list1.append(corners_1)            
        else:
            movable_list2_coor.append(coor2)
            maskd2 = np.zeros(img2.shape[:2],np.uint8)
            maskd2[int(coor2[0]):int(coor2[2]), int(coor2[1]):int(coor2[3])] = 1
            maskb2[int(coor2[0]):int(coor2[2]), int(coor2[1]):int(coor2[3])] = 0
            cornerd_2 = cv2.goodFeaturesToTrack(img2_greq, mask = maskd2, **feature_paramd)
            cornerd_1, st, err = cv2.calcOpticalFlowPyrLK(img2_greq, img1_greq, cornerd_2, None, **lk_params)          
            cornerd_2 = cornerd_2[st==1]
            movable_list2.append(cornerd_2)
            cornerd_1 = cornerd_1[st==1]
            movable_list1.append(cornerd_1)       

            
       
        cornerb_2 = cv2.goodFeaturesToTrack(img2_greq, mask = maskb2, **feature_params)
        cornerb_1, st, err = cv2.calcOpticalFlowPyrLK(img2_greq, img1_greq, cornerb_2, None, **lk_params)
        cornerb_2 = cornerb_2[st==1]
        # background_list2.append(cornerb_2)
        cornerb_1 = cornerb_1[st==1]
        # background_list1.append(cornerb_1)

        
        F,mask = computeFundamentalMatrix(cornerb_2, cornerb_1)
        
        
        #concatenate the corners points from background parts and chosen bounding box         
        #evaluate the F matrix and see the mask of inlier(1) and outlier(0)
    
        for obj_corners2, obj_corners1, coor in zip(movable_list2, movable_list1, movable_list2_coor):
            if len(obj_corners2) < 3: 
                # not enougn information for decision making, label the box as unknown in green
                c1_g, c2_g = (coor[1], coor[0]), (coor[3], coor[2])
                cv2.rectangle(img2, c1_g, c1_g, (0,255,0), 1) 
                for tmp_g in obj_corners2:
                    x2_g,y2_g = tmp_g.ravel()
                    cv2.circle(img2,(x2_g,y2_g),3,(0,255,0),-1)
                continue                            
            cornerc2 = np.concatenate((obj_corners2, cornerb_2), axis=0)
            cornerc1 = np.concatenate((obj_corners1, cornerb_1), axis=0) #cornerc -> corner concatenated
            F_esti, mask_movable = computeFundamentalMatrix(cornerc2, cornerc1)
            
            #to do
            #add the outlier of semantic constraint and FVB)Constraint (parallax of static labels)
            #further check the inliers from mask_movable 
            
            prob_moving = 1-sum(mask_movable[0:len(obj_corners2)])/len(obj_corners2)

            if prob_moving > 0.5:               
                # plot the moving object in red
                c1_r, c2_r = (coor[1], coor[0]), (coor[3], coor[2])
                cv2.rectangle(img2, c1_r, c2_r, (0,0,255), 1)                                                                
                # plot corner outliers in the moving object
                for tmp_r in obj_corners2:
                    x2_r,y2_r = tmp_r.ravel()
                    cv2.circle(img2,(x2_r,y2_r),3,(0,0,255),-1)
            else:                
                # plot static objects in blue
                c1_b, c2_b = (coor[1], coor[0]), (coor[3], coor[2])
                cv2.rectangle(img2, c1_b, c2_b, (255,0,0), 1)
                for tmp_b in obj_corners2:
                    x2_b,y2_b = tmp_b.ravel()
                    cv2.circle(img2,(x2_b,y2_b),3,(255,0,0),-1)

    #save the image into RANSAC folder
    cv2.imwrite('/home/songming/tensorflow-yolov4-tflite/Sequence_05/RANSAC/' + file_name2 + ext2, img2)
    
    
                
    
   

    

