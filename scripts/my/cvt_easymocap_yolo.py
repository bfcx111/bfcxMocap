import os
import os.path as osp
import shutil
from tqdm import tqdm
import json
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# from easymocap.mytools import read_camera
#images
#|-445+19+000852.jpg  
#|-beijia+6+000130.jpg
#labels
#文件夹名+视角+文件名


def xyxybox2yolo(bbox, width, height):
    c = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    boxsize = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    out = [c[0]/width, c[1]/height, boxsize[0]/width, boxsize[1]/height]
    for i in range(4):
        if out[i]>1:
            return "", False
    out = list(map(lambda x: '{:.6f}'.format(x), out))
    text = ' '.join(out)
    return text ,True

def get_ann_img(root, cam, annid):
    annname = join(root, 'annots2', cam, annid)
    if not os.path.exists(annname):
        return False, None, None
    with open(annname,'r') as f:
        data = json.load(f)
    # imgname = join(root, data['filename'])
    imgname = join(root, 'images', cam, annid.split('.')[0]+'.jpg')
    if not os.path.exists(imgname):
        return False, None, None
    return True, data, imgname
def checklab(bbox,threshold):
    if bbox[4]<threshold:
        return False
    # for i in range(4):
    #     if bbox[i]>1:
    #         return False

    return True

def convert_easymocap_yolo(root, out):
    images_dir=osp.join(out, 'images')
    labels_dir=osp.join(out, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    personname = root.split('/')[-1].split('_')[0]
    camlist = sorted(os.listdir(join(root, 'annots2')))
    print('camlist : ',camlist)
    for cam in tqdm(camlist):
        annlist = sorted(os.listdir(join(root, 'annots2', cam)))
        img_num=0
        for annid in annlist:
            # if img_num%10!=0:
            #     img_num+=1
            #     continue
            img_num+=1

            flag, data, image_path = get_ann_img(root, cam, annid)
            if flag ==False:
                break
            image_content=[]
            height = data['height']
            width = data['width']
            for i in range(len(data['annots'])):
                ann = data['annots'][i]
                bbox_body = ann['bbox']
                bbox_body[3] = height
                bbox_handl2d = ann['bbox_handl2d']
                bbox_handr2d = ann['bbox_handr2d']
                # bbox_face2d = ann['bbox_face2d']
                threshold = 0.25
                if checklab(bbox_body, threshold):
                    txt, Flag = xyxybox2yolo(bbox_body, width, height)
                    if Flag:
                        image_content.append(f'0 {txt}')
                if checklab(bbox_handl2d, threshold):
                    txt, Flag = xyxybox2yolo(bbox_handl2d, width, height)
                    if Flag:
                        image_content.append(f'1 {txt}')
                if checklab(bbox_handr2d, threshold):
                    txt, Flag = xyxybox2yolo(bbox_handr2d, width, height)
                    if Flag:
                        image_content.append(f'2 {txt}')
                # if checklab(bbox_face2d, threshold):
                #     image_content.append(f'3 {xyxybox2yolo(bbox_face2d, width, height)}')

            # os.makedirs(images_dir, exist_ok=True)
            outimgpath = join(images_dir,'{}+{}+{}'.format(personname, cam, image_path.split('/')[-1]))
            txt_path = join(labels_dir,'{}+{}+{}.txt'.format(personname, cam, annid.split('.')[0]))

            if not os.path.exists(outimgpath):
                shutil.copy(image_path, outimgpath)

            with open(txt_path, 'w') as f:
                f.write('\n'.join(image_content))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    convert_easymocap_yolo(args.path, args.out)
