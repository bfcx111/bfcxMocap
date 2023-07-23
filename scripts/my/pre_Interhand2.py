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
from easymocap.estimator.wrapper_base import bbox_from_keypoints
from easymocap.mytools.file_utils import read_json, save_json
import cv2
import pdb
from tqdm import tqdm
import shutil

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def check_kpts(pt, shape, d=5):
    x=pt[:,0]
    y=pt[:,1]
    v=pt[:,2]  
    ff = (((0-d)<x) & (x<(shape[1]+d)) & ((0-d)<y) &(y<(shape[0]+d)))^0
    pt[:,2]=v * ff

def get_need(split):
    root_path = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/images/'+split
    need={}
    caps = os.listdir(root_path)
    
    for cap in tqdm(caps):
        seqs = os.listdir(join(root_path,cap))

        need[cap]={}
        for seq in seqs:
            need[cap][seq]={}
            cams = os.listdir(join(root_path,cap,seq))
            for cam in cams:
                if cam[1] == '1':
                    continue
                need[cap][seq][cam]=[]
                frames = os.listdir(join(root_path,cap,seq,cam))
                for i in range(len(frames)):
                    if i%100==0:
                        # pdb.set_trace()
                        fid=frames[i].split('.')[0].split('image')[1]
                        need[cap][seq][cam].append(fid)
    # with open('/nas/users/wangjunhao/out/need.json','w') as f:
    #     json.dump(need,f)
    return need



def pre_Interhand(out, split, need):
     
    root_path = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/'
    # pdb.set_trace()
    print("load----")
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'camera')),'r') as f:
        cameras = json.load(f)

    joint_num = 21
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}

    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'joint_3d')),'r') as f:
        joints = json.load(f)

    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'data')),'r') as f:
        data = json.load(f)
    
    print("end----")
    info_ann = data['annotations']
    info_img = data['images']
    new_ann={}
    new_img={}
    for i in tqdm(range(len(info_img))):
        # if i <397000:
        #     continue
        # if i%1000!=0:
        #     continue
        ann = info_ann[i]
        img = info_img[i]

        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']
        
        if cam[1] == '1':
            continue

        if str(frame_idx) not in need['Capture'+str(capture_id)][str(seq_name)]['cam'+str(cam)]:
            continue
        
        img_path = join(root_path, 'images', split, img['file_name'])
        images = cv2.imread(img_path)
        if images.mean() < 10:
            continue
        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
        joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
        joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(joint_num*2)
        # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
        # joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
        # joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]
        #right
        ptr = np.zeros((21,3))
        # pdb.set_trace()
        ptr[0,:2] = joint_img[20,:]
        ptr[0,2] = joint_valid[20]
        idx = [3,2,1,0]
        idx = np.array(idx)
        for n in range(5):
            ptr[1+n*4:5+n*4,:2] = joint_img[idx+n*4,:]
            ptr[1+n*4:5+n*4,2] = joint_valid[idx+n*4]
        #left
        ptl = np.zeros((21,3))
        ptl[0,:2] = joint_img[41,:]
        ptl[0,2] = joint_valid[41]
        for n in range(5):
            ptl[1+n*4:5+n*4,:2] = joint_img[21+idx+n*4,:]
            ptl[1+n*4:5+n*4,2] = joint_valid[21+idx+n*4]
        check_kpts(ptl,images.shape, 5)
        check_kpts(ptr,images.shape, 5)
        out_path_img = join(out,split,'images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx))
        if not os.path.exists(os.path.dirname(out_path_img)):
            os.makedirs(os.path.dirname(out_path_img))
        
        shutil.copy(img_path,out_path_img)
        res_write={
            "filename" : join('images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx)),
            "height" : img['height'],
            "width" : img['width'],
            "annots" : [{
                "personID": 0,
                "bbox_handl2d": bbox_from_keypoints(ptl),
                "bbox_handr2d": bbox_from_keypoints(ptr),
                "handl2d": ptl.tolist(),
                "handr2d": ptr.tolist()
            }]
        }
        save_json(join(out, split, 'annots', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.json'.format(frame_idx)), res_write)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    need = get_need(args.split)
    pre_Interhand(args.out, args.split,need)
