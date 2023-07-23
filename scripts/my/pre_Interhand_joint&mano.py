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
    # pdb.set_trace()
    ff = (((0-d)<x) & (x<(shape[1]+d)) & ((0-d)<y) &(y<(shape[0]+d)))^0
    pt[:,2]=v * ff

def map_interhand_openpose(data):
    map_idx = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
    handl = np.array(data['handl3d'])[map_idx]
    handr = np.array(data['handr3d'])[map_idx]
    data['handl3d'] = handl.tolist()
    data['handr3d'] = handr.tolist()
    return data

def pre_Interhand(out, split,args):
     
    root_path = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/'
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'camera')),'r') as f:
        cameras = json.load(f)

    joint_num = 21
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}

    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'joint_3d')),'r') as f:
        joints = json.load(f)
    
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_MANO_NeuralAnnot.json'.format(split)),'r') as f:
        mano = json.load(f)

    caps = os.listdir(root_path+'images/'+split)
    save_id=0
    for cap in tqdm(caps):
        seqs = os.listdir(join(root_path+'images/'+split,cap))
        for seq in seqs:
            cams = os.listdir(join(root_path+'images/'+split,cap,seq))
            for cam_name in cams:
                cam = cam_name.split('cam')[1]
                if cam[1] == '1':
                    continue
                frames = os.listdir(join(root_path+'images/'+split,cap,seq,cam_name))
                frames.sort()
                for i in range(len(frames)):
                    if i%5==0:
                        capture_id = cap.split('Capture')[1]
                        seq_name = seq
                        frame_idx = int(frames[i].split('.')[0].split('image')[1])
                        # frame_idx = frames[i].split('.')[0].split('image')[1]
        
                        img_path = join(root_path, 'images', split, cap, seq, cam_name, frames[i])
                        images = cv2.imread(img_path)
                        if images.mean() < 10:
                            continue
                        
                        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
                        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
                        joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
                        joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]
                        joint_valid = np.array(joints[str(capture_id)][str(frame_idx)]['joint_valid'],dtype=np.float32).reshape(joint_num*2)

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

                        if(sum(ptl[:,2]==0)+sum(ptr[:,2]==0)>32):
                            continue

                        shutil.copy(img_path,out_path_img)
                        
                        K = np.eye(3)
                        K[0, 0] = focal[0]
                        K[1, 1] = focal[1]
                        K[0, 2] = princpt[0]
                        K[1, 2] = princpt[1]
                        RT = np.eye(4)
                        R = np.array(camrot)
                        T = np.array(campos) /1000
                        T = -np.dot(R, T.reshape(3,1))
                        #2d kpts
                        res_write={
                        "filename" : join('images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx)),
                        "height" : images.shape[0],
                        "width" : images.shape[1],
                        'K' : K.tolist(), 
                        "annots" : [{
                                "personID": 0,
                                "bbox_handl2d": bbox_from_keypoints(ptl),
                                "bbox_handr2d": bbox_from_keypoints(ptr),
                                "handl2d": ptl.tolist(),
                                "handr2d": ptr.tolist()
                        }]
                        }
                        #3d kpts            
                        coord = np.array(joint_cam)/1000
                        conf = np.array(joint_valid).reshape(-1,1)
                        coord[conf[:,0]<1] = 0.
                        k3d = np.hstack([coord, conf])
                        annot={
                            "handl3d": k3d[21:].tolist(),
                            "handr3d": k3d[:21].tolist()
                        }
                        annot = map_interhand_openpose(annot)
                        res_write["annots"][0].update(annot)
                        #mano
                        mano_ann={"manol":{}, "manor":{}}
                        for hand_type in ["left", "right"]:
                            mano_param = mano[str(capture_id)][str(frame_idx)][hand_type]
                            mano_pose, mano_shape, mano_trans = [[0.,0.,0.]]*16, [0.0]*10, [0.0]*3
                            if mano_param is not None:

                                mano_pose = np.array(mano_param['pose']).reshape(-1, 3).tolist()
                                mano_shape = mano_param['shape']
                                mano_trans = mano_param['trans']

                            mano_ann['mano'+hand_type[0]]={
                                "pose" : mano_pose,
                                "shape" : mano_shape,
                                "trans" : mano_trans,
                            }
                        res_write["annots"][0].update(mano_ann)

                        save_json(join(out, split, 'annots', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.json'.format(frame_idx)), res_write)
                        save_id+=1
                        if save_id>=1000:
                            return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    pre_Interhand(args.out, args.split, args)

    code_name = '/pre_Interhand_joint.py'
    # shutil.copy(os.getcwd()+'/scripts/my'+code_name,args.out+code_name)
#python scripts/my/pre_Interhand_joint.py /home/wangjunhao/data/Interhand2.6M --split='val'