'''
  @ Author: wjh
'''
import numpy as  np
from easymocap.estimator.wrapper_base import bbox_from_keypoints
from easymocap.mytools.file_utils import save_json
import json
from os.path import join
path = '/dellnas/dataset/rtmocap/0729_5/forlabel-merge'
# path = '/home/xxx/datasets/new_data'
def check_bbox(bbox,w,h):
    new_bbox = [
        max(0,bbox[0]),
        max(0,bbox[1]),
        min(w,bbox[2]),
        min(h,bbox[3]),
        bbox[4]
        ]
    return new_bbox
    
with open(join(path,'task314_coco.json'),'r') as f:
    data = json.load(f)
    
    
data_im = data['images']
data_ann =data['annotations']

all_info = {}
for info in data_im:
    dt = {
        "filename": 'images/merge/'+info['file_name'],
        "height": info['height'],
        "width": info['width'],
        "annots": [],
        "isKeyframe": False
        }
    all_info[info['id']] = dt

for info in data_ann:
    kpts = np.array(info['keypoints']).reshape(25,3)
    kpts[kpts[:,2]>0,2] = 1
    img_id = info['image_id']
    bbox = bbox_from_keypoints(kpts)
    bbox = check_bbox(bbox,all_info[img_id]['width'],all_info[img_id]['height'])
    dt = {
            "personID": len(all_info[img_id]['annots']),
            "bbox": bbox,
            "keypoints": kpts.tolist(),
            "isKeyframe": False
    }
    
    all_info[img_id]['annots'].append(dt)

for idx in all_info:
    info = all_info[idx]
    name = info['filename'].split('/')[-1].replace('.jpg','.json')
    name = name.split('%2B')
    name = 'annots/{}+{}+{}+{}+{}'.format(name[0],name[1],name[2],name[3],name[4])
    save_json(join(path,name),info)