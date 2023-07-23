#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:33:21 2022

@author: wjh
"""

import os
from os.path import join
import json
import numpy as np
from easymocap.mytools.triangulator import batch_triangulate, iterative_triangulate
from easymocap.mytools.camera_utils import read_cameras
from easymocap.annotator.file_utils import save_json
from tqdm import tqdm
from easymocap.mytools.vis_base import plot_bbox

cam_param_root = ''#'/home/xxx/DGPU/dellnas/dataset/HalfCapture/process220802-step10/colmap-align'
boxname2class={
        'bbox':'person',
        'bbox_handl2d':'handl',
        'bbox_handr2d':'handr',    
        'bbox_face2d' :'face'   
    }
def projectPoints(X, K, R, t, Kd):    
    x = R @ X + t
    x[0:2,:] = x[0:2,:]/x[2,:]#到归一化平面
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]

    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])
    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    return x

def get_box_size(personid, cam, args):
    w={
        'person' : 0,
        'handl' : 0,
        'handr' : 0,
        'face' : 0
        }
    h={
        'person' : 0,
        'handl' : 0,
        'handr' : 0,
        'face' : 0
        }
    
    # for personid in range(5):
        # personid = 1
    root = args.root
    # root = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/'.format(personid)
    # if str(personid) =="0" :
    #     path = join(root,'object2d', cam)
    # else:
    #     path = join(root,'hand-yolo2', cam)
    path = join(root, args.annot, cam)
    from easymocap.annotator.file_utils import getFileList    
    annnames = getFileList(path, ext='.json')
    for annname in annnames:
        with open(join(path, annname), 'r') as f:
            data  = json.load(f)
        ann = data['annots']
        for c_name in ['bbox','bbox_handl2d','bbox_handr2d','bbox_face2d']:
            if c_name not in ann[0].keys():
                continue
            b = ann[0][c_name]
            w[boxname2class[c_name]] = max(w[boxname2class[c_name]], b[2]-b[0])
            h[boxname2class[c_name]] = max(h[boxname2class[c_name]], b[3]-b[1])
        # for a in ann:
        #     b=a['bbox']
        #     w[a['class']] = max(w[a['class']], b[2]-b[0])
        #     h[a['class']] = max(h[a['class']], b[3]-b[1])

    box_size={}
    for name in ['person', 'handl', 'handr', 'face']:
        box_size[name] = (round(w[name]),round(h[name]))
    return box_size
def centertobox(c, name, box_size):
    
    return [
    c[0]-box_size[name][0]/2,
    c[1]-box_size[name][1]/2,
    c[0]+box_size[name][0]/2,
    c[1]+box_size[name][1]/2,
    c[2]
    ]

def test(args):


    import cv2
    personid = 0
    # root = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/'.format(personid)
    root = args.root
    path = join(root,'annots')
    cam_pam = read_cameras(root)
    camlist = sorted(os.listdir(path))
    annlist = sorted(os.listdir(join(path, camlist[0])))
    # ann_root = '/home/xxx/datasets/out/process/person{}/annots'.format(personid)
    ann_root = args.out
    L=[]
    for cam in camlist:
        box_size = get_box_size(personid, cam, args)
        K = cam_pam[cam]['K']
        K = np.matrix(K)
        R = cam_pam[cam]['R']
        R = np.matrix(R)
        t = cam_pam[cam]['T']
        Kd =cam_pam[cam]['dist'].reshape(5)
        for annid in annlist: 
            img_path = join(root, 'images', cam, annid.replace('.json', '.jpg'))    
            img = cv2.imread(img_path)
    
            ann_path = join(ann_root, annid)
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            
            
            ans={}
            for c_name in ann:    
                kpts = np.array(ann[c_name]).reshape((-1,4)).transpose()
                pt = projectPoints(kpts[0:3,:], K, R, t, Kd)
                ans[c_name]=pt.tolist()
                cv2.circle(img, (int(pt[0,0]),int(pt[1,0])), 50, (0,0,255), 10)
                bbox = centertobox([int(pt[0,0]),int(pt[1,0]),pt[2,0]], c_name, box_size)
                plot_bbox(img, bbox, c_name, scale=1)
            img = cv2.resize(img,(1024,512))
            cv2.imshow('vis',img)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                L.append(img)
                break
    if False:
        LL=[[],[],[],[]]
        imgL=[]
        for i in range(4):
            for j in range(4):
                LL[i].append(L[i*4+j])
            imgL.append(np.hstack(LL[i]))
        res = np.vstack(imgL)
        cv2.imshow('vis',res)
        k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def pre_center3d(args):
    root = args.root #'/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/'
    annot_name = args.annot #'object2d'
    out = args.out #'/home/xxx/datasets/out/person0/annots'
    # out = join(root, 'output-obj3d')
    os.makedirs(out, exist_ok=True)
    
    path = join(root,annot_name)
    camlist = sorted(os.listdir(path))
    annlist = sorted(os.listdir(join(path, camlist[0])))
    cam_pam = read_cameras(cam_param_root)
    
    Pall=[]
    # for cam_name in cam_pam.keys():
    #     RT.append(cam_pam[cam_name]['RT'])
    for cam_name in cam_pam.keys():
        Pall.append(cam_pam[cam_name]['P'])
    Pall = np.stack(Pall, axis=0)
    
    # boxname2class={
    #     'bbox':'person',
    #     'bbox_handl2d':'handl',
    #     'bbox_handr2d':'handr',        
    # }
    
    for annid in tqdm(annlist):
        data = {
            'person' : [],
            'handl' : [],
            'handr' : [],
            'face' : []
            }
        for cam in camlist:
    
            with open(join(path, cam, annid), 'r') as f:
                obj_ann  = json.load(f)
            existname = []
            # for ann in obj_ann['annots']:
            #     data[ann['class']].append(ann['keypoints'])
            #     existname.append(ann['class'])
            for box_name in ['bbox','bbox_handl2d','bbox_handr2d','bbox_face2d']:
                if box_name not in obj_ann['annots'][0].keys():
                    continue
                box = obj_ann['annots'][0][box_name]
                data[boxname2class[box_name]].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[4]])
                existname.append(boxname2class[box_name])
            for name in ['person', 'handl', 'handr', 'face']:
                if name not in existname:
                    data[name].append([[0.0, 0.0, 0.0]])
        result={}
        for c_name in data:
            data[c_name] = np.array(data[c_name]).reshape((-1,1,3))
            res = batch_triangulate(data[c_name],Pall)
            result[c_name]=res.tolist()
        save_json(join(out, annid), result)

def pre_box2d(args):
    root = args.root #'/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/'
    annot_name = args.annot #'object2d'
    
    pid={
        'person' : 0,
        'handl' : 1,
        'handr' : 2,
        'face' : 3
        }
    # personid = root.split('person')[1].split('_step60')[0]
    personid = root.split('/')[-1]
    print("personid : ",personid)
    # root = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/'.format(personid)
    path = join(root,'annots')
    cam_pam = read_cameras(cam_param_root)
    camlist = sorted(os.listdir(path))
    annlist = sorted(os.listdir(join(path, camlist[0])))
    ann_root = args.out
    # breakpoint()
    # out = "/home/xxx/datasets/out/process0628/person{}/annots".format(personid)
    # out = "/home/xxx/datasets/out/process0628/{}/bbox_2d".format(personid)
    # out = '/home/xxx/datasets/new_data/test111/{}/bbox_2d'.format(personid)
    out = root+'/bbox_2d'
    for cam in tqdm(camlist,desc='save 2dbox'):
        box_size = get_box_size(personid, cam, args)
        K = cam_pam[cam]['K']
        K = np.matrix(K)
        R = cam_pam[cam]['R']
        R = np.matrix(R)
        t = cam_pam[cam]['T']
        Kd =cam_pam[cam]['dist'].reshape(5)
        for annid in annlist: 
            ann_path = join(ann_root, annid)
            with open(ann_path, 'r') as f:
                ann = json.load(f)

            with open(join(root, annot_name, cam, annid), 'r') as f:
                data  = json.load(f)
            confidence = {
                'person':1,
                'handl':1,
                'handr':1,
                'face':1
            }
            for c_name in ['bbox','bbox_handl2d','bbox_handr2d','bbox_face2d']:
                if c_name in data['annots'][0].keys():
                    confidence[boxname2class[c_name]] = data['annots'][0][c_name][-1]
                else:
                    confidence[boxname2class[c_name]] = 0
            
            ans={}
            annots=[]
            for c_name in ann:    
                kpts = np.array(ann[c_name]).reshape((-1,4)).transpose()
                pt = projectPoints(kpts[0:3,:], K, R, t, Kd)
                pt[2,0] = kpts[3,0]
                ans[c_name]=pt.tolist()
                bbox = centertobox([int(pt[0,0]),int(pt[1,0]),confidence[c_name]], c_name, box_size)
                annots.append({
                    "personID": pid[c_name],
                    "class": c_name,
                    "bbox": bbox,
                    "keypoints": [
                      [int(pt[0,0]),int(pt[1,0]),pt[2,0]]
                    ],
                    "isKeyframe": False
                },)
                if c_name == 'face':
                    facebox = bbox

            # for i in range(len(annots)):
            #     if annots[i]['class'] == 'person':
            #         personbox = annots[i]['bbox']
            #         personbox[1] = min(personbox[1],facebox[1])
            #         personbox[3] = min(personbox[3],data['height'])
                    
            data['annots'] = annots
            outname = join(out, cam, annid)
            os.makedirs(join(out, cam), exist_ok=True)
            save_json(outname, data)
            


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
     #bbox-3d
    # parser.add_argument('personid', type=int)
    parser.add_argument('--root', default='/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/',type=str)
    parser.add_argument('--annot', default='annots', type=str) #orig file?
    parser.add_argument('--iter', default='/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/bbox_3d',action='store_true')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    print(args.root)
    args.out = join(args.root,'bbox_3d')
    # cam_param_root = os.path.abspath(join(args.root,'..','colmap-align'))
    cam_param_root = os.path.abspath(args.root)
    print(cam_param_root)
    pre_center3d(args)
    pre_box2d(args)
        
    
    

'''
python3 scripts/my/box_triangulate.py /home/xxx/datasets/out/process0628-3d/person0/annots --root /home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/ --annot object2d
python3 scripts/my/box_triangulate.py /home/xxx/datasets/out/process0628-3d/person1/annots --root /home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person1_step60/ --annot hand-yolo2
python3 scripts/my/box_triangulate.py /home/xxx/datasets/out/process0628-3d/person2/annots --root /home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person2_step60/ --annot hand-yolo2
python3 scripts/my/box_triangulate.py /home/xxx/datasets/out/process0628-3d/person3/annots --root /home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person3_step60/ --annot hand-yolo2
python3 scripts/my/box_triangulate.py /home/xxx/datasets/out/process0628-3d/person4/annots --root /home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person4_step60/ --annot hand-yolo2


python3 scripts/my/box_triangulate_easymocap.py /home/xxx/datasets/new_data/test111/action_yyk/bbox_3d --root /home/xxx/datasets/new_data/test111/action_yyk 
python3 scripts/my/box_triangulate_easymocap.py /dellnas/dataset/HalfCapture/process220802-step10/action_zdh/bbox_3d --root /dellnas/dataset/HalfCapture/process220802-step10/action_zdh
python scripts/my/check_easymocap_joint.py /home/xxx/datasets/new_data/test111/action_zdh --yolo_box --annot bbox_2d

python3 scripts/my/box_triangulate_easymocap.py --root /dellnas/dataset/HalfCapture/process220802-step10/action_gyl


for data in /dellnas/dataset/HalfCapture/process220802-step10/action_*
python3 scripts/my/box_triangulate_easymocap.py --root ${data}


/dellnas/dataset/HalfCapture/process220802-step10/action_zsy


python3 scripts/my/box_triangulate_easymocap.py --root /dellnas/dataset/rtmocap/0728+001000+001380
'''

