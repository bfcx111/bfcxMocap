'''
  @ Author: wjh
'''

from os.path import join
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.estimator.wrapper_base import bbox_from_keypoints
from easymocap.mytools.file_utils import read_json, save_json
import cv2
import pdb
from tqdm import tqdm
import shutil




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=str)
    args = parser.parse_args()
    root = '/home/wangjunhao/data/InterHand_step5/{}/'.format(args.split)
    out = '/nas/users/wangjunhao/data/hand/{}/'.format(args.split)
    ann_out = out + 'annots/'
    img_out = out + 'images/'
    ann_path = root + 'annots/'
    img_path = root + 'images/'
    L = os.listdir(ann_path)
    # import pdb;pdb.set_trace()
    for name in tqdm(L):
      # print(ann_path+name, ann_out+name)
      # print(img_path+name.split('.')[0]+'.jpg', img_out+name.split('.')[0]+'.jpg')
      # break
      shutil.copy(ann_path+name, ann_out+name)
      shutil.copy(img_path+name.split('.')[0]+'.jpg', img_out+name.split('.')[0]+'.jpg')

#python scripts/my/pre_Interhand.py /home/wangjunhao/data/Interhand2.6M --split='train'