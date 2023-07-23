import json
import numpy as np
import cv2
import os
from easymocap.mytools.file_utils import save_json
from easymocap.estimator.wrapper_base import bbox_from_keypoints

import shutil
from tqdm import tqdm
from os.path import join

def get_annots(ann,hand_type):
    bbox = bbox_from_keypoints(np.array(ann['hand_pts']))
    # bbox = np.array(ann['head_box']).reshape(-1).tolist()
    if hand_type == 'left':
        return {
            "bbox_handl2d":bbox,
            "handl2d":ann['hand_pts']
            }
    else :
        return {
            "bbox_handr2d":bbox,
            "handr2d":ann['hand_pts']
            }

def pre_hand143_labels(out,split):
    path = '/home/shuaiqing/dataset_cache/hand143/hand_labels/manual_'+split
    # path = '/home/xxx/datasets/hand143/hand_labels/manual_test'
    files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    res_ann={}
    img_path_D = {}
    for name in tqdm(files, desc='converting hand_labels'+split):
        ann_path = join(path, name)
        img_path = ann_path.split('.json')[0]+'.jpg'
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        image = cv2.imread(img_path)
        # filename = name.split('.json')[0][:-2]
        namelist = name.split('_')
        if len(namelist)==3:
            imgname = namelist[0]
            person_id = int(namelist[1])
        elif len(namelist)>=4:
            imgname = name.split('_'+namelist[-1])[0]
            person_id = 0
        else:
            print("error   "+name)
        hand_type = 'right'
        if ann['is_left'] == 1:
            hand_type='left'
        if imgname not in res_ann.keys() :
            img_path_D[imgname] = img_path
            res_ann[imgname]={
                "height" : image.shape[0],
                "width" : image.shape[1],
                # "filename" : 'images/'+filename+'.jpg',
                "filename" : 'images/'+imgname+'.jpg',
                "annots" :{person_id:{
                    "personID": person_id,
                    "bbox_handl2d":[],
                    "bbox_handr2d":[],
                    "handl2d":[],
                    "handr2d":[]
                    
                }}
            }
            res_ann[imgname]['annots'][person_id].update(get_annots(ann, hand_type))        
        elif person_id not in res_ann[imgname]['annots'].keys():
            res_ann[imgname]['annots'][person_id] = {
                    "personID": person_id,
                    "bbox_handl2d":[],
                    "bbox_handr2d":[],
                    "handl2d":[],
                    "handr2d":[] 
                }
            res_ann[imgname]['annots'][person_id].update(get_annots(ann, hand_type))
        else:
            res_ann[imgname]['annots'][person_id].update(get_annots(ann, hand_type))


    os.makedirs(join(out, 'annots'), exist_ok=True)
    os.makedirs(join(out, 'images'), exist_ok=True)
    for name in tqdm(res_ann, desc='saving hand_labels'+split):
        shutil.copy(img_path_D[name], join(out, res_ann[name]['filename']))
        annlist=[]
        for ai in res_ann[name]['annots']:
            annlist.append(res_ann[name]['annots'][ai])
        res_ann[name]['annots']=annlist
        save_json(join(out, 'annots', name+'.json'), res_ann[name])
    
if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('out', type=str)
    # parser.add_argument('split', type=str)

    # args = parser.parse_args()
    # path = '/home/xxx/datasets/hand143/hand_labels/manual_test'
    # out = '/nas/users/wangjunhao/data/hand143/handlabels/'
    out = '/home/wangjunhao/data/hand143/lab_out/'
    split = 'test'
    pre_hand143_labels(out+split,split)
    split = 'train'
    pre_hand143_labels(out+split,split)