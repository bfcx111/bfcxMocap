import os
import os.path as osp
import shutil
from tqdm import tqdm
import json
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def xyxybox2yolo(bbox, width, height):
    c = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    boxsize = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    out = [c[0]/width, c[1]/height, boxsize[0]/width, boxsize[1]/height]
    out = list(map(lambda x: '{:.6f}'.format(x), out))
    text = ' '.join(out)
    return text

def get_ann_img(root, cam, annid):
    annname = join(root, 'annots', cam, annid)
    if not os.path.exists(annname):
        return False, None, None
    with open(annname,'r') as f:
        data = json.load(f)
    imgname = join(root, data['filename'])
    if not os.path.exists(imgname):
        return False, None, None
    return True, annname, imgname
def convert_easymocap_yolo(root):
    
    personname = root.split('/')[-1].split('_')[0]
    out = '/dellnas/dataset/HalfCapture/process0617/yolo0622/'+personname+'_step60'
    images_dir=osp.join(out, 'images')
    # annots_dir=osp.join(out, 'annots')
    # os.makedirs(annots_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    camlist = sorted(os.listdir(join(root, 'images')))
    print('camlist : ',camlist)
    for cam in tqdm(camlist):
        imglist = sorted(os.listdir(join(root, 'images', cam)))
        img_num=0
        for imgid in imglist:
            if img_num%60!=0:
                img_num+=1
                continue
            img_num+=1

            image_path = join(root, 'images', cam, imgid)
            # flag, ann_path, image_path = get_ann_img(root, cam, annid)
            # if flag ==False:
            #     break

            outimgpath = join(images_dir, cam, '{}'.format(image_path.split('/')[-1]))
            # outannpath = join(annots_dir, cam, '{}'.format(ann_path.split('/')[-1]))
            os.makedirs(join(images_dir, cam), exist_ok=True)
            # os.makedirs(join(annots_dir, cam), exist_ok=True)
            shutil.copy(image_path, outimgpath)
            # shutil.copy(ann_path, outannpath)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    # parser.add_argument('out', type=str)
    args = parser.parse_args()    

    convert_easymocap_yolo(args.path)
