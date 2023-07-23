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
    bbox[0] = max(bbox[0] ,0)
    bbox[1] = max(bbox[1] ,0)
    bbox[2] = min(bbox[2] ,width)
    bbox[3] = min(bbox[3] ,height)
    if bbox[1]>bbox[3] or bbox[0]>bbox[2]:
        return "", False
    c = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    boxsize = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    out = [c[0]/width, c[1]/height, boxsize[0]/width, boxsize[1]/height]
    for i in range(4):
        if out[i]>1:
            return "", False
    out = list(map(lambda x: '{:.6f}'.format(x), out))
    text = ' '.join(out)
    return text ,True

def get_ann_img(root, cam, annid, args):
    annname = join(root, args.annot, cam, annid)
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

def convert_easymocap_yolo(root, out, args):
    images_dir=osp.join(out, 'images')
    labels_dir=osp.join(out, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    personname = root.split('/')[-1].split('_')[0]
    camlist = sorted(os.listdir(join(root, args.annot)))
    print('camlist : ',camlist)
    for cam in tqdm(camlist):
        annlist = sorted(os.listdir(join(root, args.annot, cam)))
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
                class_name = ann['class']
                bbox = ann['bbox']

                threshold = 0.25
                txt, Flag = xyxybox2yolo(bbox, width, height)
                if class_name == 'person' and checklab(bbox, threshold) and Flag:
                        image_content.append(f'0 {txt}')
                elif class_name == 'handl' and checklab(bbox, threshold) and Flag:
                        image_content.append(f'1 {txt}')
                elif class_name == 'handr' and checklab(bbox, threshold) and Flag:
                        image_content.append(f'2 {txt}')
                elif class_name == 'face' and checklab(bbox, threshold) and Flag:
                        image_content.append(f'3 {txt}')

            # os.makedirs(images_dir, exist_ok=True)
            outimgpath = join(images_dir,'{}+{}+{}'.format(personname, cam, image_path.split('/')[-1]))
            txt_path = join(labels_dir,'{}+{}+{}.txt'.format(personname, cam, annid.split('.')[0]))

            if not os.path.exists(outimgpath):
                print("error : "+outimgpath)
                shutil.copy(image_path, outimgpath)

            with open(txt_path, 'w') as f:
                f.write('\n'.join(image_content))


from easymocap.annotator.file_utils import getFileList 
def convert_class_yolo2(root, out, args):
    def get_ann_img2(root, annid, args):
        annname = join(root, args.annot, annid)
        if not os.path.exists(annname):
            return False, None, None
        with open(annname,'r') as f:
            data = json.load(f)
        # imgname = join(root, data['filename'])
        imgname = join(root, args.image_name, annid.replace('.json','.jpg'))# 'images'
        if not os.path.exists(imgname):
            return False, None, None
        return True, data, imgname
    images_dir=osp.join(out, 'images')
    labels_dir=osp.join(out, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    person_name = root.split('/')[-1]
    # breakpoint()
    # personname = root.split('/')[-1].split('_')[0]
    # camlist = sorted(os.listdir(join(root, 'annots')))
    # print('camlist : ',camlist)
    # for cam in tqdm(camlist):
    # annlist = sorted(os.listdir(join(root, args.annot)))
    annlist = getFileList(join(root, args.annot), ext='.json')
    img_num=0
    for annid in tqdm(annlist):
        # if img_num%10!=0:
        #     img_num+=1
        #     continue
        img_num+=1

        flag, data, image_path = get_ann_img2(root, annid, args)
        if flag ==False:
            break
        image_content=[]
        height = data['height']
        width = data['width']
        for i in range(len(data['annots'])):
            ann = data['annots'][i]
            class_name = ann['class']
            bbox = ann['bbox']

            threshold = 0.25
            txt, Flag = xyxybox2yolo(bbox, width, height)
            if class_name == 'person' and checklab(bbox, threshold) and Flag:
                    image_content.append(f'0 {txt}')
            elif class_name == 'handl' and checklab(bbox, threshold) and Flag:
                    image_content.append(f'1 {txt}')
            elif class_name == 'handr' and checklab(bbox, threshold) and Flag:
                    image_content.append(f'2 {txt}')
            elif class_name == 'face' and checklab(bbox, threshold) and Flag:
                    image_content.append(f'3 {txt}')

        if len(image_content)<3:
            continue
        # os.makedirs(images_dir, exist_ok=True)
        
        if len(image_path.split('/'))>2:
            cam_name = image_path.split('/')[-2]
        else:
            cam_name = 'c'

        outimgpath = join(images_dir,'{}_{}_{}'.format(person_name, cam_name, image_path.split('/')[-1]))
        txt_path = join(labels_dir,'{}_{}_{}.txt'.format(person_name, cam_name, annid.split('/')[-1].split('.')[0]))

        # if False and not os.path.exists(outimgpath):
        #     print("error : "+outimgpath)
        #     break
        #     shutil.copy(image_path, outimgpath)
        if not os.path.exists(outimgpath):
                # print("error : "+outimgpath)
            shutil.copy(image_path, outimgpath)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(image_content))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--annot', default = 'annots', type=str)
    parser.add_argument('--image_name', default = 'images', type=str)

    parser.add_argument('--havecam', action='store_true')
    args = parser.parse_args()
    
    if args.havecam:
        convert_easymocap_yolo(args.path, args.out)
    else:
        convert_class_yolo2(args.path, args.out, args)
'''
root=/home/xxx/datasets/out/process0628/person0
out=/home/xxx/datasets/out/process0628/yolo
python3 scripts/my/cvt_easymocap_yolo_class.py ${root} ${out}

root=/home/xxx/datasets/out/process0628/person4
out=/home/xxx/datasets/out/process0628/yolo_val

inv :

root=/home/xxx/datasets/out/process0628/inv-yolo
out=/home/xxx/datasets/out/process0628/inv-yolo
python3 scripts/my/cvt_easymocap_yolo_class.py ${root} ${root}

root=/home/xxx/datasets/out/process0628/inv-yolo_val


root=/home/xxx/datasets/new_data/test111/action_zdh
out=/home/xxx/datasets/new_data/test111/0805

python3 scripts/my/cvt_easymocap_yolo_class.py ${root} ${out} bbox_2d

out=/dellnas/dataset/HalfCapture/yolo_0805
for data in /dellnas/dataset/HalfCapture/process220802-step10/action_*
python3 scripts/my/cvt_easymocap_yolo_class.py ${data} ${out} --annot bbox_2d


out=/dellnas/dataset/HalfCapture/yolo_0807_inv
for data in /dellnas/dataset/HalfCapture/process220802-step10/action_*
python3 scripts/my/cvt_easymocap_yolo_class.py ${data} ${out} --annot annots_inv --image_name images_inv

'''