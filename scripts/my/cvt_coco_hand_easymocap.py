'''
  @ Author: wjh
'''

import os
import os.path as osp
import shutil
from tqdm import tqdm
import json
from os.path import join
import numpy as np
from easymocap.mytools.file_utils import save_json
from easymocap.dataset.config import coco17tobody25
def check_box(b,d=2500):
    if b[2]*b[3]>d:
        return False
    return True
def check_joint(ptl, ptr, d=11):
    # if(type(ptl[2]) is int):
        
    vl = np.array(ptl).reshape(-1,3)[:,2]
    vr = np.array(ptr).reshape(-1,3)[:,2]
    use_l = True
    use_r = True
    if(type(ptl[2]) is int):
        if(sum(vl >1.5)<d):
            use_l = False
    else:
        if sum(vl >0.1)<d:
            use_l = False
    if(type(ptr[2]) is int):
        if(sum(vr >1.5)<d):
            use_r = False
    else:
        if sum(vr >0.1)<d:
            use_r = False
    # if sum(vl >0.4)>=d or sum(vr >0.4)>=d:
    #     return False
    if use_l or use_r:
        return False
    return True

def cvt_coco_body25(body,foot):
    pt = coco17tobody25(np.array(body).reshape(-1,3))
    pt[19:,:] = np.array(foot).reshape(-1,3)
    return pt

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
def convert_coco_yolo(root_path, out, split, args):
    json_path = osp.join(root_path, 'annotations', f'coco_wholebody_{split}_v1.0.json')
    with open(json_path) as f:
        json_data = json.load(f)
    info = json_data['info']  # ['description', 'url', 'version', 'year', 'date_created']
    licenses = json_data['licenses']
    images = json_data['images']  # list, ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
    annotations = json_data['annotations']  # list, ['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 
                                            # 'category_id', 'id', 'face_box', 'lefthand_box', 'righthand_box', 'lefthand_kpts', 
                                            # 'righthand_kpts', 'face_kpts', 'face_valid', 'lefthand_valid', 'righthand_valid', 'foot_valid', 'foot_kpts']
    categories = json_data['categories']  # dict_keys(['supercategory', 'id', 'name', 'keypoints', 'skeleton'])

    # update images with empty string
    imgs = {}
    res = {}
    for image in images:
        image.update({'content': []})
        imgs[image['id']] = image
        res[image['id']] = {
            "height" : image['height'],
            "width" : image['width'],
            "filename" : image['file_name'],
            "annots" :[] 
        }

    # extract bbox of each annotation into images
    for annotation in tqdm(annotations, desc='converting'):
        if annotation['iscrowd']:
            continue
        
        if args.check_hand:
            if annotation['lefthand_valid']==False and annotation['lefthand_valid']==False:
                continue
            if  check_box(annotation['lefthand_box']) and check_box(annotation['righthand_box']):
                continue
            if  check_joint(annotation['lefthand_kpts'], annotation['righthand_kpts']):
                continue
        
        
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        face_box = annotation['face_box']
        lefthand_box = annotation['lefthand_box']
        righthand_box = annotation['righthand_box']
        face_valid = annotation['face_valid']
        lefthand_valid = annotation['lefthand_valid']
        righthand_valid = annotation['righthand_valid']
        image = imgs[image_id]
        height = image['height']
        width = image['width']
        # import pdb;pdb.set_trace()
        for id1,id2 in ((0,2),(1,3)):
            lefthand_box[id2] += lefthand_box[id1]
            righthand_box[id2] += righthand_box[id1]
            bbox[id2] += bbox[id1]

        person_id = len(res[image_id]['annots'])
        ptl = np.array(annotation['lefthand_kpts']).reshape(-1,3)
        ptr = np.array(annotation['righthand_kpts']).reshape(-1,3)
        ptl[ptl[:,2]>1,2]=1
        ptr[ptr[:,2]>1,2]=1

        

        
        #置信度小于0.1的点直接置0
        ptl[ptl[:,2]<0.03,2]=0
        ptr[ptr[:,2]<0.03,2]=0
        # # import pdb;pdb.set_trace()
        # if ptl[0,2] == 0:
        #     ptl[0,:] = np.array(annotation['keypoints']).reshape(-1,3)[9,:]
        # if ptr[0,2] == 0:
        #     ptr[0,:] = np.array(annotation['keypoints']).reshape(-1,3)[10,:]
        
        lefthand_box += [1.0]
        righthand_box += [1.0]
        if args.check_hand:
            if check_box(annotation['lefthand_box']):
                ptl[:,:]=0
                lefthand_box = [0,0,100,100,0]
            if check_box(annotation['righthand_box']):
                ptr[:,:]=0
                righthand_box = [0,0,100,100,0]
            if(bbox_iou(lefthand_box, righthand_box)>0.9 or dis_kpts(ptl,ptr)<10):
                continue
            res[image_id]['annots'].append({
                "personID" : person_id,
                "bbox_handl2d" : lefthand_box,
                "bbox_handr2d" : righthand_box,
                "handl2d" : ptl.tolist(),
                "handr2d" : ptr.tolist()
            })
        else:
            body = annotation['keypoints']
            foot = annotation['foot_kpts']
            pt = cvt_coco_body25(body,foot)
            if sum(pt[:,2]<1)>24:
                continue
            res[image_id]['annots'].append({
                "personID" : person_id,
                "bbox" : bbox +[1.0],
                "keypoints" : pt.tolist()
            })

    # save
    annot_dir = osp.join(out, split, 'annots')
    images_dir = osp.join(out, split, 'images')
    os.makedirs(annot_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    num=0
    for i in tqdm(res.keys(), desc='writing'):
        # import pdb;pdb.set_trace()
        if len(res[i]['annots']) == 0:
            continue
        num+=1
        if args.save:
            file_name = res[i]['filename']
            res[i]['filename'] = 'images/coco+'+file_name
            out_ann_name = join(annot_dir,'coco+'+file_name.split('.')[0]+'.json')
            save_json(out_ann_name, res[i])
            
            imgname = join(root_path, 'images', split+'2017', file_name)
            outname = join(images_dir, 'coco+'+file_name)
            shutil.copy(imgname, outname)

            # file_name = res[i]['filename']
            # for kk in range(5):
            #     res[i]['filename'] = 'images/coco+{}+'.format(kk)+file_name
            #     out_ann_name = join(annot_dir,'coco+{}+'.format(kk)+file_name.split('.')[0]+'.json')
            #     save_json(out_ann_name, res[i])
            #     imgname = join(root_path, 'images', split+'2017', file_name)
            #     outname = join(images_dir, 'coco+{}+'.format(kk)+file_name)
            #     shutil.copy(imgname, outname)
    print(num)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to coco dataset')
    parser.add_argument('out', type=str, help='path to coco dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--check_hand', action='store_true')
    args = parser.parse_args()

    # convert_coco_yolo(args.path, 'train')
    convert_coco_yolo(args.path, args.out, 'val', args)
    convert_coco_yolo(args.path, args.out, 'train', args)
    # TODO: 考虑标注的关键点，关键点太少的不要使用
    #python scripts/my/cvt_coco_hand_easymocap.py /nas/datasets/COCO17 /nas/users/wangjunhao/data/coco --save --check_hand
    #python scripts/my/cvt_coco_hand_easymocap.py /nas/datasets/COCO17 /nas/users/wangjunhao/data/hand --save --check_hand