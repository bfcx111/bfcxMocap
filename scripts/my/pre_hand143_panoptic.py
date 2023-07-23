import json
import numpy as np
import cv2
import os
from easymocap.mytools.file_utils import save_json
from easymocap.estimator.wrapper_base import bbox_from_keypoints

import shutil
from tqdm import tqdm
from os.path import join


# path = '/home/xxx/datasets/hand143/hand143_panopticdb/hands_v143_14817.json'
path  ='/home/shuaiqing/dataset_cache/hand143/hand143_panopticdb/'
out = '/home/wangjunhao/data/hand143/hand143_panopticdb/'
with open(path+'hands_v143_14817.json', 'r') as f:
    data = json.load(f)
    
data= data['root']
os.makedirs(join(out, 'annots'), exist_ok=True)
os.makedirs(join(out, 'images'), exist_ok=True)
for ann in data:
    pt = ann['joint_self']
    bbox = bbox_from_keypoints(np.array(pt))
    res_ann={
        "height" : ann['img_height'],
        "width" : ann['img_width'],
        "filename" : 'images/'+ann['img_paths'].split('/')[1],
        "annots" :[{
            "personID": 0,
            "bbox_handl2d":[],
            "bbox_handr2d":bbox,
            "handl2d":[],
            "handr2d":pt
            
        }] 
    }
    img_id = ann['img_paths'].split('/')[1].split('.')[0]
    shutil.copy(path+ann['img_paths'], join(out, res_ann['filename']))
    save_json(join(out,'annots',img_id+'.json'), res_ann)
