#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:23:54 2022
#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1f1aa9c59fe59c90cca685b724f4f97f76137224/src/openpose/hand/handDetector.cpp#L9
@author: wjh
"""
import numpy as np
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
import json
import os
import cv2
from os.path import join

import torch
from torchvision.transforms import transforms
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.mytools.utils import Timer
from easymocap.estimator.topdown_wrapper import BaseTopDownHeatmapWrapper
from easymocap.estimator.wrapper_base import bbox_from_keypoints
from easymocap.annotator.file_utils import save_annot

def load_handnet(ckpt, device):
    from easymocap.config import load_object, Config
    # cfg = Config.load('/nas/users/wangjunhao/EasyMocapPublic/config/easypose/network/hrnet_twohand.yml')
    cfg = Config.load('/home/xxx/work/EasyMocapPublic/config/easypose/network/hrnet_twohand.yml')
    model = load_object(cfg.network_module, cfg.network_args)
    return BaseTopDownHeatmapWrapper(model, ckpt, device, input_size=(320, 320), output_size=(64, 64), use_normalize=True, name='handnet')

def rescalebox(box,rescale=1.2):
    c = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    w = (box[2]-box[0]) * rescale
    h = (box[3]-box[1]) * rescale
    bbox=[
        c[0]-w/2,
        c[1]-h/2,
        c[0]+w/2,
        c[1]+h/2,
        box[4]
        ]
    return bbox

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
        return handRectangle
    except:
        print("error getHandFromPoseIndexes")
        return {}

# def trackHand(currentRectangle, previousHands):
#     if len(previousHands)==0:
#         return currentRectangle
    
#     try:
#         prevRectangle = previousHands#[maxIndex];
#         ratio = 2;
#         newWidth = max((currentRectangle['width'] * ratio + prevRectangle['width']) * 0.5,
#                                       (currentRectangle['height'] * ratio + prevRectangle['height']) * 0.5);
#         currentRectangle['x'] = 0.5 * (currentRectangle['x'] + prevRectangle['x'] + 0.5 * (currentRectangle['width'] + prevRectangle['width']) - newWidth);
#         currentRectangle['y'] = 0.5 * (currentRectangle['y'] + prevRectangle['y'] + 0.5 * (currentRectangle['height'] + prevRectangle['height']) - newWidth);
#         currentRectangle['width'] = newWidth;
#         currentRectangle['height'] = newWidth;
#         return currentRectangle
#     except:
#         pass
        
# def detectHandsBox(kpts, mHandLeftPrevious, mHandRightPrevious):
#     kpts = np.array(kpts)
#     left=[7,6,5]
#     right= [4,3,2]
#     threshold = 0.03
#     wrist, elbow, shoulder = left
#     handRectangle_L = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
#     handRectangle_L = trackHand(handRectangle_L, mHandLeftPrevious);
#     wrist, elbow, shoulder = right
#     handRectangle_R = getHandFromPoseIndexes(kpts, wrist, elbow, shoulder, threshold)
#     handRectangle_R = trackHand(handRectangle_R, mHandRightPrevious);
#     lbbox = [
#             handRectangle_L['x'],
#             handRectangle_L['y'],
#             handRectangle_L['x']+handRectangle_L['width'],
#             handRectangle_L['y']+handRectangle_L['height'],
#             1.0
#             ]
#     rbbox = [
#             handRectangle_R['x'],
#             handRectangle_R['y'],
#             handRectangle_R['x']+handRectangle_R['width'],
#             handRectangle_R['y']+handRectangle_R['height'],
#             1.0
#             ]
#     return lbbox, rbbox

def detectHandsBox2(kpts):
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




def bbox_iou(bbox_pre, bbox_now):
    area_now = (bbox_now[2] - bbox_now[0])*(bbox_now[3]-bbox_now[1])
    area_pre = (bbox_pre[2] - bbox_pre[0])*(bbox_pre[3]-bbox_pre[1])
    # compute IOU
    # max of left
    xx1 = max(bbox_now[0], bbox_pre[0])
    yy1 = max(bbox_now[1], bbox_pre[1])
    # min of right
    xx2 = min(bbox_now[0+2], bbox_pre[0+2])
    yy2 = min(bbox_now[1+2], bbox_pre[1+2])
    # w h
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    over = (w*h)/(area_pre+area_now-w*h)
    return over

def mergebox(lb, rb):
    return [
            min(lb[0], rb[0]),
            min(lb[1], rb[1]),
            max(lb[2], rb[2]),
            max(lb[3], rb[3]),
            (lb[4]+rb[4])/2
        ]
def detectHandsKpts(results, hand_net, frame):
    # if 'handl' in results.keys() or 'handr' in results.keys(): 
    if 'handl' not in results.keys() and 'handr' not in results.keys(): 
        return np.zeros((2,42,3)),{'handl':[0,0,100,100,0], 'handr':[0,0,100,100,0]}
    if 'handl' not in results.keys():
        # results['handl'] = [0, 0, 100, 100, 0]
        results['handl'] = results['handr']
    if 'handr' not in results.keys():
        # results['handr'] = [0, 0, 100, 100, 0]
        results['handr'] = results['handl']
        
    # print("iou:",bbox_iou(results['handl'], results['handr']))
    # if  bbox_iou(results['handl'], results['handr'])>0.4:
    if False:
        bbox = mergebox(results['handl'], results['handr'])
        pts = hand_net([frame], [[{'bbox': bbox, 'rot': 0, 'fliplr': False}, {'bbox': bbox, 'rot': 0, 'fliplr': False}]])
        # res_box = {'handl':bbox, 'handr':bbox}
    else:
        pts = hand_net([frame], [[{'bbox': results['handl'], 'rot': 0, 'fliplr': False}, {'bbox': results['handr'], 'rot': 0, 'fliplr': False}]])
        # res_box = {'handl':results['handl'], 'handr':results['handr']}
        
    return pts[0][:21,:], pts[1][21:,:]
    # return pts, res_box

# def main(hand_ckpt):
def main(root, hand_ckpt):
    device = torch.device('cuda')
    hand_net = load_handnet(hand_ckpt, device)

    def get_ann_img(root, cam, annid):
        annname = join(root, 'annots', cam, annid)
        if not os.path.exists(annname):
            return False, None, None
        with open(annname,'r') as f:
            data = json.load(f)
        imgname = join(root, data['filename'])
        if not os.path.exists(imgname):
            return False, None, None
        return True, data, cv2.imread(imgname)
        

    camlist = sorted(os.listdir(join(root, 'annots')))
    for cam in camlist:
        annlist = sorted(os.listdir(join(root, 'annots', cam)))
        for annid in annlist:
            flag, data, frame = get_ann_img(root, cam, annid)
            if flag ==False:
                break
            for i in range(len(data['annots'])):
                ann = data['annots'][i]
                
                if 'keypoints' not in ann.keys():
                    print("[error] not keypoints in {}".format(annid))
                    break
                #gethandbox
                lbbox, rbbox = detectHandsBox2(ann['keypoints'])
                res={'handl': lbbox,'handr': rbbox}
                ptl, ptr = detectHandsKpts(res, hand_net, frame)
        
                lbbox = bbox_from_keypoints(ptl, rescale=1.68)
                rbbox = bbox_from_keypoints(ptr, rescale=1.68)
                while_num=1
                while while_num>0: #False: #
                    while_num-=1
                    res={'handl': lbbox,'handr': rbbox}
                    ptl, ptr = detectHandsKpts(res, hand_net, frame)
                    lbbox = bbox_from_keypoints(ptl, rescale=1.68)
                    rbbox = bbox_from_keypoints(ptr, rescale=1.68)
                
                data['annots'][i].update({
                    "bbox_handl2d": lbbox,
                    "bbox_handr2d": rbbox,
                    "handl2d": ptl.tolist(),
                    "handr2d": ptr.tolist()
                    })
            outpath = join(root,'annots2', cam)
            os.makedirs(outpath, exist_ok=True)
            outname = join(outpath, annid)
            save_annot(outname, data)
            


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    hand_ckpt = '/home/xxx/DGPU/dellnas/users/shuaiqing/cache-easypose-6/hrnet+hand+320x320/model/last.ckpt'
    main(root=args.path, hand_ckpt=hand_ckpt)
