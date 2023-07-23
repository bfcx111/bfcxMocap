'''
  @ Date: 2021-12-11 23:05:58
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-04-07 21:59:56
  @ FilePath: /EasyMocapPublic/apps/preprocess/extract_object2d.py
'''
from easymocap.annotator.file_utils import getFileList, read_json, save_annot, save_json
import torch
from os.path import join
import cv2
import numpy as np
import os
from tqdm import tqdm

def find_circle(src):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=20, 
        param1=300, param2=0.9, minRadius=img.shape[0]//6, maxRadius=img.shape[0]//2)
    cimg = src.copy()
    if circles is not None and circles.shape[0] > 0: # Check if circles have been found and only then iterate     over these and add them to the image
        circles = np.uint16(np.around(circles))
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3,     cv2.LINE_AA)  # draw center of circle
        return True, (circles[0][0][0], circles[0][0][1], circles[0][0][2])
    else:
        return False, (-1, -1, -1)

def extract_object(path, out, imgnames, ckpt):
    model = torch.hub.load('ultralytics/yolov5', 'custom', ckpt)
    for basename in tqdm(imgnames):
        imgname = join(path, 'images', basename)
        annname = join(path, 'annots', basename.replace('.jpg', '.json'))
        annots = read_json(annname)
        outname = imgname.replace('images', out).replace('.jpg', '.json')
        if os.path.exists(outname):
            continue
        results = model(imgname)
        arrays = np.array(results.pandas().xyxy[0])
        results = []
        img = cv2.imread(imgname)
        breakpoint()
        for pid, res in enumerate(arrays):
            bbox = {
                'personID': pid,
                'class': res[6],
                'bbox': [res[0], res[1], res[2], res[3], res[4]],
            }
            breakpoint()
            cx = (res[0] + res[2])/2
            cy = (res[1] + res[3])/2
            bbox['keypoints'] = [[cx, cy, res[4]]]
            if False: # find the circle
                l, t, r, b = res[:4]
                size = int(max(r-l, b-t)/2) * 2
                center = [(l+r)/2, (t+b)/2]
                l = max(int(center[0]-size), 0)
                r = min(int(center[0]+size), img.shape[1])
                t = max(int(center[1]-size), 0)
                b = min(int(center[1]+size), img.shape[0])
                crop = img[t:b, l:r]
                flag, (cx, cy, radius) = find_circle(crop)
                if flag:
                    cx += l
                    cy += t
                    bbox['center'] = [int(cx), int(cy)]
                    bbox['radius'] = int(radius)
            results.append(bbox)
        annots['annots'] = results
        save_annot(outname, annots)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--out', type=str, help="the output of data", 
        default='object')
    parser.add_argument('--ext', type=str, 
        default='jpg', choices=['jpg', 'png'], 
        help="image file extension")
    parser.add_argument('--ckpt', type=str, 
        default='data/models/yolov5-basketball.pt', 
        help="the path of ckpt")
    args = parser.parse_args()

    imgnames = getFileList(join(args.path, 'images'), ext=args.ext)
    imgnames.sort(key=lambda x:int(x.split(os.sep)[-1].split('.')[0]))
    extract_object(args.path, args.out, imgnames, args.ckpt)