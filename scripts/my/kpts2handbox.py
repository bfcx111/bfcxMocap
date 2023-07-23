#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:23:54 2022
#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1f1aa9c59fe59c90cca685b724f4f97f76137224/src/openpose/hand/handDetector.cpp#L9
@author: xxx
"""
import numpy as np
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
import json
import os
import cv2
#from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, coco17tobody25, get_affine_transform, get_final_preds


# colors_bar_rgb = [
#     (94, 124, 226), # 青色
#     (255, 200, 87), # yellow
#     (74,  189,  172), # green
#     (8, 76, 97), # blue
#     (219, 58, 52), # red
#     (77, 40, 49), # brown
# ]

# colors_table = {
#     'b': [0.65098039, 0.74117647, 0.85882353],
#     '_pink': [.9, .7, .7],
#     '_mint': [ 166/255.,  229/255.,  204/255.],
#     '_mint2': [ 202/255.,  229/255.,  223/255.],
#     '_green': [ 153/255.,  216/255.,  201/255.],
#     '_green2': [ 171/255.,  221/255.,  164/255.],
#     'r': [ 251/255.,  128/255.,  114/255.],
#     '_orange': [ 253/255.,  174/255.,  97/255.],
#     'y': [ 250/255.,  230/255.,  154/255.],
#     'g':[0,255/255,0],
#     'k':[0,0,0],
#     '_r':[255/255,0,0],
#     '_g':[0,255/255,0],
#     '_b':[0,0,255/255],
#     '_k':[0,0,0],
#     '_y':[255/255,255/255,0],
#     'purple':[128/255,0,128/255],
#     'smap_b':[51/255,153/255,255/255],
#     'smap_r':[255/255,51/255,153/255],
#     'person': [255/255,255/255,255/255],
#     'handl': [255/255,51/255,153/255],
#     'handr': [51/255,255/255,153/255],
# }

# def get_rgb(index):
#     if isinstance(index, int):
#         if index == -1:
#             return (255, 255, 255)
#         if index < -1:
#             return (0, 0, 0)
#         # elif index == 0:
#         #     return (245, 150, 150)
#         col = list(colors_bar_rgb[index%len(colors_bar_rgb)])[::-1]
#     else:
#         col = colors_table.get(index, (1, 0, 0))
#         col = tuple([int(c*255) for c in col[::-1]])
#     return col
# def plot_bbox(img, bbox, pid, scale=1, vis_id=True):
#     # 画bbox: (l, t, r, b)
#     x1, y1, x2, y2, c = bbox
#     if c < 0.01:return img
#     x1 = int(round(x1*scale))
#     x2 = int(round(x2*scale))
#     y1 = int(round(y1*scale))
#     y2 = int(round(y2*scale))
#     color = get_rgb(pid)
#     lw = max(img.shape[0]//300, 2)
#     cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
#     if vis_id:
#         font_scale = img.shape[0]/1000
#         cv2.putText(img, '{}'.format(pid), (x1, y1+int(25*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def getDistance(keypoints, elementA, elementB):
    try:
        keypointPtr = keypoints
        # keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
        pixelX = keypointPtr[elementA,0] - keypointPtr[elementB,0];
        pixelY = keypointPtr[elementA,1] - keypointPtr[elementB,1];
        return np.sqrt(pixelX*pixelX+pixelY*pixelY);
    except:
        print("error getDistance")
        return -1
def getHandFromPoseIndexes(poseKeypoints, wrist, elbow, shoulder, threshold):
    try:
        
        handRectangle={
            'x':0,
            'y':0,
            'width':0,
            'height':0
        }
        posePtr = poseKeypoints
        # posePtr = &poseKeypoints.at(person*poseKeypoints.getSize(1)*poseKeypoints.getSize(2));
        wristScoreAbove = (posePtr[wrist,2] > threshold);
        elbowScoreAbove = (posePtr[elbow,2] > threshold);
        shoulderScoreAbove = (posePtr[shoulder,2] > threshold);
        ratioWristElbow = 0.33;
        if wristScoreAbove and elbowScoreAbove and shoulderScoreAbove:
            handRectangle['x'] = posePtr[wrist,0] + ratioWristElbow * (posePtr[wrist,0] - posePtr[elbow,0]);
            handRectangle['y'] = posePtr[wrist,1] + ratioWristElbow * (posePtr[wrist,1] - posePtr[elbow,1]);
            distanceWristElbow = getDistance(poseKeypoints, wrist, elbow);
            distanceElbowShoulder = getDistance(poseKeypoints, elbow, shoulder);
            handRectangle['width'] = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder);
    
        handRectangle['height'] = handRectangle['width']
        # x-y refers to the center --> offset to topLeft point
        handRectangle['x'] -= handRectangle['width'] / 2;
        handRectangle['y'] -= handRectangle['height'] / 2;
        # bbox = [handRectangle['x'],handRectangle['y'],
        #         handRectangle['x']+handRectangle['width'],handRectangle['y']+handRectangle['height']]
        return handRectangle #bbox #
    except:
        print("error getHandFromPoseIndexes")
        return {}


def trackHand(currentRectangle, previousHands):
    if len(previousHands)==0:
        return currentRectangle
    
    try:
        prevRectangle = previousHands#[maxIndex];
        ratio = 2;
        newWidth = max((currentRectangle['width'] * ratio + prevRectangle['width']) * 0.5,
                                      (currentRectangle['height'] * ratio + prevRectangle['height']) * 0.5);
        currentRectangle['x'] = 0.5 * (currentRectangle['x'] + prevRectangle['x'] + 0.5 * (currentRectangle['width'] + prevRectangle['width']) - newWidth);
        currentRectangle['y'] = 0.5 * (currentRectangle['y'] + prevRectangle['y'] + 0.5 * (currentRectangle['height'] + prevRectangle['height']) - newWidth);
        currentRectangle['width'] = newWidth;
        currentRectangle['height'] = newWidth;
        return currentRectangle
        
    except:
        pass
        
def detectHands(kpts):
    kpts = np.array(kpts)
    left=[7,6,5]
    right= [4,3,2]
    threshold = 0.03
    wrist, elbow, shoulder = left
    handRectangle_L = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
    handRectangle_L = trackHand(handRectangle_L, mHandLeftPrevious);
    wrist, elbow, shoulder = right
    handRectangle_R = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
    handRectangle_R = trackHand(handRectangle_R, mHandRightPrevious);
    lbbox = [
            handRectangle_L['x'],
            handRectangle_L['y'],
            handRectangle_L['x']+handRectangle_L['width'],
            handRectangle_L['y']+handRectangle_L['height'],
            1.0
            ]
    rbbox = [
            handRectangle_R['x'],
            handRectangle_R['y'],
            handRectangle_R['x']+handRectangle_R['width'],
            handRectangle_R['y']+handRectangle_R['height'],
            1.0
            ]
    return lbbox, rbbox

def detectHands2(kpts):
    kpts = np.array(kpts)
    left=[7,6,5]
    right= [4,3,2]
    threshold = 0.03
    wrist, elbow, shoulder = left
    handRectangle_L = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
    wrist, elbow, shoulder = right
    handRectangle_R = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
    lbbox = [
            handRectangle_L['x'],
            handRectangle_L['y'],
            handRectangle_L['x']+handRectangle_L['width'],
            handRectangle_L['y']+handRectangle_L['height'],
            1.0
            ]
    rbbox = [
            handRectangle_R['x'],
            handRectangle_R['y'],
            handRectangle_R['x']+handRectangle_R['width'],
            handRectangle_R['y']+handRectangle_R['height'],
            1.0
            ]
    return lbbox, rbbox

def getKeypointsRectangle(keypoints, threshold):
    keypoints = np.array(keypoints)
    try:
        v = keypoints[:,2]
        x = keypoints[:,0]
        y = keypoints[:,1]
        v = v>threshold
        minX, maxX = x[v].min(),x[v].max()
        minY, maxY = y[v].min(),y[v].max()
        return {'x':minX, 'y':minY, 'width':maxX-minX, 'height':maxY-minY}
    except:
        return {}

    
def updateTracker(handKeypoints):
    try:
        thresholdRectangle = 0.25
        scoreThreshold = 0.66667
        
        handLeftRectangle = getKeypointsRectangle(handKeypoints[0], thresholdRectangle)
        handRightRectangle = getKeypointsRectangle(handKeypoints[1], thresholdRectangle)
        mHandLeftPrevious = handLeftRectangle
        mHandRightPrevious = handRightRectangle
        return mHandLeftPrevious, mHandRightPrevious
#        if :
#            mHandLeftPrevious = handLeftRectangle
#        if (handRightRectangle.area() > 0):
#            mHandRightPrevious = handRightRectangle
            
    except:
        print("error updateTracker")

def gethandbox(keypoints):
    lbbox, rbbox = detectHands(keypoints)

path = '/home/xxx/DGPU/nas/dataset/HalfCapture/process0614/220614-0_4+001100+010600/'
split = '02/'
ann_path = path+ 'annots/'+split
img_path = path+'images/'+split
path = 'C:/Users/29506/Desktop/test/'
ann_path = 'C:/Users/29506/Desktop/test/annots/'
img_path = 'C:/Users/29506/Desktop/test/images/'


L= os.listdir(ann_path)
mHandLeftPrevious={}
mHandRightPrevious={}




#for name in L:
for i in range(len(L)):
    name = L[i]
    #getbodykeypoints
    with open(ann_path+name,'r') as f:
        data = json.load(f)
    ann = data['annots'][0]
    #gethandbox
    lbbox, rbbox = detectHands(ann['keypoints']) 
    lbbox2, rbbox2 = detectHands2(ann['keypoints'])
    #gethandkeypoints
    ptl = ann['handl2d']
    ptr = ann['handr2d']
    #update previous hand box
    mHandLeftPrevious, mHandRightPrevious = updateTracker([ptl,ptr])
    
    if i==0 :
        with open(ann_path+L[0],'r') as f:
            data = json.load(f)
        ann = data['annots'][0]
        lbbox, rbbox = detectHands(ann['keypoints'])
        ptl = ann['handl2d']
        ptr = ann['handr2d']
        mHandLeftPrevious, mHandRightPrevious = updateTracker([ptl,ptr])
        
    img = cv2.imread(img_path+name.split('.')[0]+'.jpg')

    vis = img.copy()
    
    plot_bbox(vis,rbbox2,0)
    plot_bbox(vis,lbbox2,1)
    
    plot_bbox(img,rbbox,0)
    plot_bbox(img,lbbox,1)
    
#    img = cv2.resize(img,(512,512))
#    vis = cv2.resize(vis,(512,512))
    cv2.imshow('vis',img)
    cv2.imshow('vis2',vis)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()