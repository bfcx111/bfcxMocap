#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:06:23 2022

@author: wjh
"""
import os
from os.path import join
import cv2
from tqdm import tqdm
import json
from easymocap.annotator.file_utils import getFileList, read_json, save_annot, save_json

# num=0
# root = '/home/xxx/datasets/out/process0628/'
root = '/dellnas/dataset/HalfCapture/process220802-step10'
# out = '/home/xxx/datasets/out/process0628/inv-yolo'
ext_cam_id = '02'
# os.makedirs(join(out, 'images_inv'), exist_ok=True)
images_name = 'images'
annot_name = 'bbox_2d'

file_list_tmp = os.listdir(root)
file_list = []

for name in file_list_tmp:
    if 'action' in name:
        file_list.append(name)

# os.makedirs(join(root, 'images_inv'), exist_ok=True)
# file_list = ['action_yyk']
# for personid in tqdm(range(4), desc='img'):
for name in tqdm(file_list, desc='img'):
    path = join(root, name, images_name)
    camlist = sorted(os.listdir(path))
    imglist = sorted(os.listdir(join(path, camlist[0])))
    os.makedirs(path.replace(images_name,'images_inv'), exist_ok=True)
    for cam in camlist:
        for imgid in imglist: 
            # breakpoint()
            if not os.path.exists(join(path.replace(images_name,'images_inv'),'inv+{}+{}+{}').format(name, cam, imgid)):
                img = cv2.imread(join(path, cam, imgid))
                img = cv2.flip(img, 1)
                cv2.imwrite(join(path.replace(images_name,'images_inv'),'inv+{}+{}+{}').format(name, cam, imgid),img)
            
            # if False:
            #     img =cv2.resize(img,(1024,512))
            #     cv2.imshow('vis',img)
            #     k = cv2.waitKey(0) & 0xFF
            #     if k == ord('q'):
            #         break
    #         num+=1
    #         if num==1000:
    #             break
    #     if num==1000:
    #         break
    # if num==1000:
    #     break

# cv2.destroyAllWindows()
# num=0
# os.makedirs(join(out, 'annots_inv'), exist_ok=True)
# os.makedirs(join(root, 'annots_inv'), exist_ok=True)
# for personid in tqdm(range(4), desc='ann'):
for name in tqdm(file_list, desc='ann'):
    path = join(root, name, annot_name)
    camlist = sorted(os.listdir(path))
    annlist = sorted(os.listdir(join(path, camlist[0])))
    for cam in camlist:
        for annid in annlist:
            ann_path = join(path, cam, annid)
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            for j in range(len(ann['annots'])):
                box = ann['annots'][j]['bbox']
                box[0], box[2] = ann['width']-box[2], ann['width']-box[0]
                ann['annots'][j]['bbox'] = box
                kpts = ann['annots'][j]['keypoints']
                kpts[0][0] = ann['width']-kpts[0][0]
                ann['annots'][j]['keypoints'] = kpts
                if ann['annots'][j]['class']=='handl':
                    ann['annots'][j]['class']='handr'
                    ann['annots'][j]['personID']=2
                elif ann['annots'][j]['class']=='handr':
                    ann['annots'][j]['class']='handl'
                    ann['annots'][j]['personID']=1    
            save_json(join(path.replace(annot_name,'annots_inv'),'inv+{}+{}+{}').format(name, cam, annid), ann)
    #         num+=1
    #         if num==1000:
    #             break
    #     if num==1000:
    #         break
    # if num==1000:
    #     break



# root = '/home/xxx/datasets/out/process0628/'
# out = '/home/xxx/datasets/out/process0628/inv-yolo_val'
# os.makedirs(join(out, 'images'), exist_ok=True)
# for personid in tqdm([4], desc='img_val'):
#     path = join(root, 'person{}'.format(personid), 'images')
#     camlist = sorted(os.listdir(path))
#     imglist = sorted(os.listdir(join(path, camlist[0])))
#     for cam in camlist:
#         for imgid in imglist: 
#             if not os.path.exists(join(out,'images','inv+person{}+{}+{}').format(personid, cam, imgid)):
#                 img = cv2.imread(join(path, cam, imgid))
#                 img = cv2.flip(img, 1)
#                 cv2.imwrite(join(out,'images','inv+person{}+{}+{}').format(personid, cam, imgid),img)


# os.makedirs(join(out, 'annots'), exist_ok=True)
# for personid in [4]:
#     path = join(root, 'person{}'.format(personid), 'annots')
#     camlist = sorted(os.listdir(path))
#     annlist = sorted(os.listdir(join(path, camlist[0])))
#     for cam in tqdm(camlist, desc='ann'):
#         for annid in annlist:
#             ann_path = join(path, cam, annid)
#             with open(ann_path, 'r') as f:
#                 ann = json.load(f)
#             for j in range(len(ann['annots'])):
#                 box = ann['annots'][j]['bbox']
#                 box[0], box[2] = ann['width']-box[2], ann['width']-box[0]
#                 ann['annots'][j]['bbox'] = box
#                 kpts = ann['annots'][j]['keypoints']
#                 kpts[0][0] = ann['width']-kpts[0][0]
#                 ann['annots'][j]['keypoints'] = kpts
#                 if ann['annots'][j]['class']=='handl':
#                     ann['annots'][j]['class']='handr'
#                     ann['annots'][j]['personID']=2
#                 elif ann['annots'][j]['class']=='handr':
#                     ann['annots'][j]['class']='handl'
#                     ann['annots'][j]['personID']=1    
#             save_json(join(out,'annots','inv+person{}+{}+{}').format(personid, cam, annid), ann)


# if False:
#     import shutil
#     root = '/home/xxx/datasets/out/process0628/'
#     out = '/home/xxx/datasets/out/process0628/yolo'
#     os.makedirs(join(out, 'annots'), exist_ok=True)
#     for personid in tqdm(range(4), desc='ann'):
#         path = join(root, 'person{}'.format(personid), 'annots')
#         camlist = sorted(os.listdir(path))
#         annlist = sorted(os.listdir(join(path, '02')))
#         for cam in camlist:
#             for annid in annlist:
#                 ann_path = join(path, cam, annid)

#                 shutil.copy(ann_path, join(out,'annots','person{}+{}+{}').format(personid, cam, annid))

#     out = '/home/xxx/datasets/out/process0628/yolo_val'
#     os.makedirs(join(out, 'annots'), exist_ok=True)
#     for personid in tqdm([4], desc='ann'):
#         path = join(root, 'person{}'.format(personid), 'annots')
#         camlist = sorted(os.listdir(path))
#         annlist = sorted(os.listdir(join(path, '02')))
#         for cam in camlist:
#             for annid in annlist:
#                 ann_path = join(path, cam, annid)
#                 shutil.copy(ann_path, join(out,'annots','person{}+{}+{}').format(personid, cam, annid))

'''
for data in /dellnas/dataset/HalfCapture/process220802-step10/action_*

'''