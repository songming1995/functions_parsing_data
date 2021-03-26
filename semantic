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
# params for ShiTomasi corner detection
# feature_paramd = dict( maxCorners = 5,
#                        qualityLevel = 0.4,
#                        minDistance = 3,
#                        )

# feature_params = dict( maxCorners = 150,
#                        qualityLevel = 0.5,
#                        minDistance = 3,
#                        )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 3,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# def computeFundamentalMatrix(kps_ref, kps_cur):             
#     F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, 2, 0.98)
#     if F is None or F.shape == (1, 1):
#         # no fundamental matrix found
#         raise Exception('No fundamental matrix found')
#     elif F.shape[0] > 3:
#         # more than one matrix found, just pick the first
#         print('more than one matrix found, just pick the first')
#         F = F[0:3, 0:3]
#     return np.matrix(F), mask

# /home/songming/tensorflow-yolov4-tflite/Sequence_05/images -> image directory 05
# /home/songming/tensorflow-yolov4-tflite/Sequence_05/labels -> label directory 05
# /home/songming/tensorflow-yolov4-tflite/Sequence_071/images -> image directory 071
# /home/songming/tensorflow-yolov4-tflite/Sequence_071/labels -> label directory 071
# /home/songming/tensorflow-yolov4-tflite/data/classes/coco.names -> semantic label directory

 

# image_list = os.listdir('/home/songming/tensorflow-yolov4-tflite/Sequence_071/images')
image_list = os.listdir('/home/songming/tensorflow-yolov4-tflite/Sequence_05/images')
image_list.sort()
res = []
img_array = []

class_file_name = '/home/songming/tensorflow-yolov4-tflite/data/classes/coco.names'
names = {}  # dictionary to save the class label
with open(class_file_name, 'r') as data:
    for ID, name in enumerate(data):
        names[ID] = name.strip('\n')

num_classes = len(names)
static_object = [9, 10, 11, 12, 13]


for k in range(0,len(image_list)):
# for k in range(0,1):
    
    
    file_name, ext = os.path.splitext(image_list[k]) #split filename and format
    
    
    img1 = cv2.imread(os.path.join('/home/songming/tensorflow-yolov4-tflite/Sequence_05/images', image_list[k]))
    image_h, image_w, _ = img1.shape    
    box = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name + 'box.npy')    
    score = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name  + 'score.npy')        
    classes = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name + 'classes.npy')    
    valid = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_05/labels' +'/' + file_name + 'valid.npy')
    
    # img1 = cv2.imread(os.path.join('/home/songming/tensorflow-yolov4-tflite/Sequence_071/images', image_list[k]))
    # image_h, image_w, _ = img1.shape    
    # box = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_071/labels' +'/' + file_name + 'box.npy')    
    # score = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_071/labels' +'/' + file_name  + 'score.npy')        
    # classes = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_071/labels' +'/' + file_name + 'classes.npy')    
    # valid = np.load('/home/songming/tensorflow-yolov4-tflite/Sequence_071/labels' +'/' + file_name + 'valid.npy')
    
    # write the images to a video    
    # size = (image_w, image_h)
    # img_array.append(img1) 
    
    
    for i in range(valid[0]):        
        coor = np.copy(box[0][i])    
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        fontScale = 0.5
        class_ind = int(classes[0][i])
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        #define box color, blue (255,0,0) for static , green (0,255,0) for unknown, red (0,0,255) for dynamic
        if class_ind in static_object:
            bbox_color = (255,0,0) #blue stands for static object
        else:
            bbox_color = (0,255,0) #green stands for unknown object, needs geometric verification
        
        cv2.rectangle(img1, c1, c2, bbox_color, bbox_thick)
        bbox_mess = '%s:' % (names[class_ind])
        #bbox_mess = '%s: %.2f' % (names[class_ind], score[0][i])
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img1, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
    
        cv2.putText(img1, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        # cv2.imwrite('/home/songming/kitti001.png', img1)
        # cv2.imwrite('/home/songming/tensorflow-yolov4-tflite/Sequence_071/imagesL/' + file_name + ext , img1)
        
        
       
# write the images to a video        
# out = cv2.VideoWriter('05.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()




    # cv2.imwrite('/home/songming/001/kitti001.png', img1)
    
    # pred_bbox = detector(img1)

    # box = pred_bbox[0]
    # score = pred_bbox[1]
    # classes = pred_bbox[2]
    # valid = pred_bbox[3]
    

    
    # np.save('/home/songming/tensorflow-yolov4-tflite/image_071/labels/' + file_name + 'box.npy', box)
    # np.save('/home/songming/tensorflow-yolov4-tflite/image_071/labels/' + file_name + 'score.npy', score)
    # np.save('/home/songming/tensorflow-yolov4-tflite/image_071/labels/' + file_name + 'classes.npy', classes)
    # np.save('/home/songming/tensorflow-yolov4-tflite/image_071/labels/' + file_name + 'valid.npy', valid)
    
    
    
    # pred_bbox = []
    # pred_bbox.append(box)
    # pred_bbox.append(score)
    # pred_bbox.append(classes)
    # pred_bbox.append(valid)    
    

    # draw the bbox and label on the images
    # original_image = cv2.imread(os.path.join('image_05', image_list[k]))
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # image = utils.draw_bbox(original_image, pred_bbox)
    # image = Image.fromarray(image.astype(np.uint8))
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imwrite('/home/songming/Documents/yolo/image_05/' + file_name + ext , image)
    
# def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
#     num_classes = len(classes)
#     image_h, image_w, _ = image.shape
#     hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
#     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#     colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
#     random.seed(0)
#     random.shuffle(colors)
#     random.seed(None)
    
#     out_boxes, out_scores, out_classes, num_boxes = bboxes
#     for i in range(num_boxes[0]):
#         if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
#         coor = out_boxes[0][i]
#         coor[0] = int(coor[0] * image_h)
#         coor[2] = int(coor[2] * image_h)
#         coor[1] = int(coor[1] * image_w)
#         coor[3] = int(coor[3] * image_w)
    
#         fontScale = 0.5
#         score = out_scores[0][i]
#         class_ind = int(out_classes[0][i])
#         bbox_color = colors[class_ind]
#         bbox_thick = int(0.6 * (image_h + image_w) / 600)
#         c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
#         cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    
#         if show_label:
#             bbox_mess = '%s: %.2f' % (classes[class_ind], score)
#             t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
#             c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
#             cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
    
#             cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
#     return image





            
    # pred_bbox = detector(img2)
    # box55 = pred_bbox[0]
    # score55 = pred_bbox[1]
    # classes55 = pred_bbox[2]
    # valid55 = pred_bbox[3]
    
    # image_h, image_w, _ = img1.shape
    
    
    # # gloabl mask for static parts of the scene
    # masks = np.ones(img1.shape[:2],np.uint8)
    
    # global list container for corners in movable objects
    # save the object's index in the list during semantic RANSAC
    # instance-level object is saved as an array in the list
    # for each instance, it is composed by corner points in the numpy array
    # movable_list1 = []
    # movable_list2 = []
    
    # # global list container for corners in static objects
    # static_list1 = []
    # static_list2 = []
    
    # # global list container for corners in the background
    # background_list1 = []
    # background_list2 = []
    
    # #pay attention to the gloabl and local variable with the same name
    # for i in range(valid50[0]):        
    #     coor = np.copy(box50[0][i])    
    #     coor[0] = int(coor[0] * image_h)
    #     coor[2] = int(coor[2] * image_h)
    #     coor[1] = int(coor[1] * image_w)
    #     coor[3] = int(coor[3] * image_w)
    #     masks[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] = 0
    #     if score50[0][i] < 0.4: # threshold to filter out low probability bboxes
    #         continue
    #     c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])        
    #     cv2.rectangle(img1, c1, c2, (255,0,0), 1)
    #     maskd = np.zeros(img1.shape[:2],np.uint8)
    #     maskd[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] = 1
        
    #     corners_1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), mask = maskd, **feature_paramd)
    #     corners_2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
    #                                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners_1, None, **lk_params)
    #     corners_1 = corners_1[st==1]
    #     movable_list1.append(corners_1)
    #     corners_2 = corners_2[st==1]
    #     movable_list2.append(corners_2)
        
    #     for corner in corners_1:
    #         x,y = corner.ravel()
    #         cv2.circle(img1,(x,y),3,(0,0,255),-1)    
    
    #     for corner2 in corners_2:
    #         x2,y2 = corner2.ravel()
    #         cv2.circle(img2,(x2,y2),3,(0,0,255),-1)   
            
    # for i in range(valid55[0]):
    #     if score55[0][i] < 0.4:
    #         continue
    #     coor = np.copy(box55[0][i])
    #     coor[0] = int(coor[0] * image_h)
    #     coor[2] = int(coor[2] * image_h)
    #     coor[1] = int(coor[1] * image_w)
    #     coor[3] = int(coor[3] * image_w)
    #     c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])        
    #     cv2.rectangle(img2, c1, c2, (255,0,0), 1)
        
        
    #exclude the bounding box area to find static background points for F matrix estimation       
    # if (1):   #use local parameters, no conflict with others
    #     corners_1 = cv2.goodFeaturesToTrack(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), mask = masks, **feature_params)
    #     corners_2, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
    #                                        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners_1, None, **lk_params)
    #     corners_1 = corners_1[st==1]
    #     background_list1.append(corners_1)
    #     corners_2 = corners_2[st==1]
    #     background_list2.append(corners_2)    
    
    #     for corner in corners_1:
    #         x,y = corner.ravel()
    #         cv2.circle(img1,(x,y),3,(255,0,0),-1)    # blue corner points drawing in prev frame
    
    #     for corner2 in corners_2:
    #         x2,y2 = corner2.ravel()
    #         cv2.circle(img2,(x2,y2),3,(255,0,0),-1)  # blue corner points drawing in curr frame
            
    #     F,mask = computeFundamentalMatrix(corners_1, corners_2)
        
    #     for obj_corners1, obj_corners2 in zip(movable_list1, movable_list2):
    #         corners1 = np.concatenate((obj_corners1, corners_1), axis=0)
    #         corners2 = np.concatenate((obj_corners2, corners_2), axis=0)
    #         F_esti, mask_movable = computeFundamentalMatrix(corners1, corners2)
    #         a = 1-sum(mask_movable[0:len(obj_corners1)])/len(obj_corners1)
    #         res.append(a)
    #         # Epipolar equation x'Fx=0
            
            
            
        
    #concatenate the corners points from background parts and chosen bounding box, evaluate the F matrix and 
    #see the mask of inlier(1) and outlier(0), save the number of background points, the rest are on the movable
    #objects 
       
    #find corner points in the bounding boxes and track these points to next frame
    
    #1. Define the ROI img1[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])] according to the bbox coordinantes
    
    #2. Find good features to tarck with KLT in the ROI
    
    #3. Calculate the opipolar geometry redisudals with the established correspondence points
    
    #4. Decision-making for moving objects based on the outliers proportion in the boundong box
    
    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    # img = cv2.add(frame,mask)
    
    
    # t = time.time()
        #run a backward-check for match verification between frames, lk_track.py, ensure the bounding box feature
        #points have correspondence
        # p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        # d = abs(p0-p0r).reshape(-1, 2).max(-1)
        # status = d < back_threshold
    # computingtime = time.time() - t
    
#     img1 = cv2.resize(img1, (0, 0), None, 1, 1)
#     img2 = cv2.resize(img2, (0, 0), None, 1, 1)
#     numpy_horizontal = np.vstack((img1, img2))
#     #numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)    
#     cv2.imshow('KLT tracking',numpy_horizontal)
    

