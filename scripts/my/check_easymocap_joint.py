'''
  @ Author: wjh
'''

from os.path import join
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
import cv2
import pdb
from tqdm import tqdm


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
    if(area_pre+area_now-w*h)<=0:
        return 0
    over = (w*h)/(area_pre+area_now-w*h)
    return over
def dis_kpts(l,r):
    # import pdb;pdb.set_trace()
    l = np.array(l)
    r = np.array(r)
    v = l[:,2]*r[:,2]
    v[v>0]=1
    if(sum(v)==0):
        return 999999
    d = abs(l[:,:2]-r[:,:2])
    d = (sum(d[:,0]*v)+sum(d[:,1]*v))/sum(v)
    return d

def vis_hand(root,args):
    ann_path = join(root, 'annots')
    img_path = join(root, 'images')
    L = os.listdir(ann_path)
    L.sort()
    name_id=0
    all_file_num=len(L)
    for name in tqdm(L):
        with open(join(ann_path,name),'r') as f:
            ann = json.load(f)
        # import pdb;pdb.set_trace()
        img = cv2.imread(join(img_path,name.split('.')[0]+'.jpg'))
        vis = img.copy()
        for a in ann['annots']:
            if args.clip:
                lb = a['bbox_handl2d']
                rb = a['bbox_handr2d']
                visl = vis.copy()[int(lb[1]):int(lb[3])+1, int(lb[0]):int(lb[2])+1]
                ptl = np.array(a['handl2d']).reshape(-1,3)
                ptl[:,0] -= int(lb[0])
                ptl[:,1] -= int(lb[1])
                visr = vis.copy()[int(rb[1]):int(rb[3])+1, int(rb[0]):int(rb[2])+1]
                ptr = np.array(a['handr2d']).reshape(-1,3)
                ptr[:,0] -= int(rb[0])
                ptr[:,1] -= int(rb[1])
                
                plot_keypoints_auto(visl, ptl, 4, lw=1)
                plot_keypoints_auto(visr, ptr, 4, lw=1)
                cv2.imshow('visl', visl)
                cv2.imshow('visr', visr)
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'):
                    break 
            # if(bbox_iou(a['bbox_handl2d'],a['bbox_handr2d'])>0.89 or dis_kpts(a['handl2d'],a['handr2d'])<10):
            #     print(bbox_iou(a['bbox_handl2d'],a['bbox_handr2d']),dis_kpts(a['handl2d'],a['handr2d']))
            
            
            if len(a['handl2d'])>0:
                plot_bbox(vis, a['bbox_handl2d'], 4, scale=1)
                plot_keypoints_auto(vis, a['handl2d'], 4, lw=1)
            if len(a['handr2d'])>0:
                plot_bbox(vis, a['bbox_handr2d'], 4, scale=1)
                plot_keypoints_auto(vis, a['handr2d'], 4, lw=1)
                

            # if len(a['handl2d'])>0 and len(a['handr2d']):

            #     pt = np.concatenate((a['handl2d'],a['handr2d']),axis=0)
            #     plot_keypoints_auto(vis,pt , 4, lw=1)
            # elif len(a['handl2d'])>0:
            #     plot_keypoints_auto(vis, a['handl2d'], 4, lw=1)
            # elif len(a['handr2d'])>0:
            #     plot_keypoints_auto(vis, a['handr2d'], 4, lw=1)
            # plot_keypoints_auto(vis, a['keypoints'], 4, lw=1)
        # cv2.imshow('vis{}/{}'.format(name_id,all_file_num), vis)
        cv2.imshow('vis', vis)
        name_id+=1
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break    
    cv2.destroyAllWindows()
    
def vis_body(root,args):
    ann_path = join(root, 'annots')
    img_path = join(root, 'images')
    L = os.listdir(ann_path)
    L.sort()
    for name in L:
        with open(join(ann_path,name),'r') as f:
            ann = json.load(f)
        # import pdb;pdb.set_trace()
        img = cv2.imread(join(img_path,name.split('.')[0]+'.jpg'))
        vis = img.copy()
        for a in ann['annots']:
            if args.clip:
                bbox = a['bbox']
                visb = vis.copy()[int(bbox[1]):int(bbox[3])+1, int(bbox[0]):int(bbox[2])+1]
                pt = np.array(a['keypoints']).reshape(-1,3)
                pt[:,0] -= int(bbox[0])
                pt[:,1] -= int(bbox[1])

                
                plot_keypoints_auto(visb, pt, 4, lw=1)
                cv2.imshow('visb', visb)
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'):
                    break         

            plot_bbox(vis, a['bbox'], 4, scale=1)
            plot_keypoints_auto(vis, a['keypoints'], 4, lw=1)
        cv2.imshow('vis', vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break    
    cv2.destroyAllWindows()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    # parser.add_argument('out', type=str)
    # parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--hand2d', action='store_true')
    parser.add_argument('--body', action='store_true')

    args = parser.parse_args()
    if args.hand2d:
        vis_hand(args.path,args)
    if args.body:
        vis_body(args.path,args)
# python scripts/my/check_easymocap_joint.py /nas/users/wangjunhao/data/coco/val