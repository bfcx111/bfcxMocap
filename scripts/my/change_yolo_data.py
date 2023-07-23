import os
import os.path as osp
import shutil
from tqdm import tqdm
import json
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from easymocap.annotator.file_utils import getFileList, read_json, save_annot, save_json

#用Yolo检测出的box可能含有多个重复的，需要去重
#这里yolo检测不到face的，要额外加上

def get_ann(body, face, cam, annid):
    body_annname = join(body, cam, annid)
    face_annname = join(face, cam, annid)
    # if not os.path.exists(annname):
    #     return False, None, None
    with open(body_annname,'r') as f:
        data_body = json.load(f)
    with open(face_annname,'r') as f:
        data_face = json.load(f)
    return True, data_body, data_face


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('personID', type=int)
    # parser.add_argument('out', type=str)
    args = parser.parse_args()
    out = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/hand-yolo2'.format(args.personID)
    path = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/hand-yolo'.format(args.personID)
    face = '/home/xxx/datasets/yolo0627/person{}_step60/annots'.format(args.personID)
    # face = '/home/xxx/DGPU/dellnas/dataset/HalfCapture/process0617/yolo0622/person{}_step60/face'.format(args.personID)
    os.makedirs(out, exist_ok=True)

    camlist = sorted(os.listdir(path))
    # camlist = ['26']
    print('camlist : ',camlist)
    for cam in tqdm(camlist):
        annlist = sorted(os.listdir(join(path, cam)))
        os.makedirs(join(out, cam), exist_ok=True)
        for annid in annlist:
            body_hand_ann, face_ann = get_ann(path, face, cam, annid)

            Lbody=[]
            Lhandl=[]
            Lhandr=[]
            for ai in range(len(body_hand_ann['annots'])):
                if body_hand_ann['annots'][ai]['class'] == "person":
                    if len(Lbody)>0 and Lbody[0]['bbox'][-1]<body_hand_ann['annots'][ai]['bbox'][-1]:
                        Lbody = [body_hand_ann['annots'][ai]]
                    else:
                        Lbody.append(body_hand_ann['annots'][ai])
                elif body_hand_ann['annots'][ai]['class'] == "handl":
                    if len(Lhandl)>0 and Lhandl[0]['bbox'][-1]<body_hand_ann['annots'][ai]['bbox'][-1]:
                        Lhandl = [body_hand_ann['annots'][ai]]
                    else:
                        Lhandl.append(body_hand_ann['annots'][ai])
                        # Lhandl.append(ai)
                elif body_hand_ann['annots'][ai]['class'] == "handr":
                    if len(Lhandr)>0 and Lhandr[0]['bbox'][-1]<body_hand_ann['annots'][ai]['bbox'][-1]:
                        Lhandr = [body_hand_ann['annots'][ai]]
                    else:
                        Lhandr.append(body_hand_ann['annots'][ai])
                        # Lhandr.append(ai)
            
            new_ann=[]
            if len(Lbody)>0:
                new_ann.append(Lbody[0])
            if len(Lhandl)>0:
                new_ann.append(Lhandl[0])
            if len(Lhandr)>0:
                new_ann.append(Lhandr[0])
            body_hand_ann['annots']=new_ann

            face_box = face_ann['annots'][0]['bbox_face2d']
            # face_box = face_ann['annots'][0]['bbox']
            new_face = {
                'personID': 4,
                'class': "face",
                'bbox': face_box,
                "keypoints": [
                    [(face_box[0]+face_box[2])/2, (face_box[1]+face_box[3])/2, face_box[4]]
                ],
                "isKeyframe": false
            }
            body_hand_ann['annots'].append(new_face)
            outname = join(out, cam, annid)
            save_annot(outname, body_hand_ann)

