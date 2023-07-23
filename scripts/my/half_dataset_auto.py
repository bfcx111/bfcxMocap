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
from easymocap.mytools.utils import Timer
from easymocap.mytools.debug_utils import run_cmd
def mv_orig_data(orig_path, out_path):
  L = os.listdir(orig_path)
  os.makedirs(out_path, exist_ok=True)
  for cam in tqdm(L, desc='mv videos'):
    videos = os.listdir(join(orig_path,cam))
    if len(videos) ==0:
      continue
    # breakpoint()
    videos = sorted(videos)
    for video in videos:
      shutil.copy(join(orig_path, cam, video),join(out_path, '{}_{}'.format(cam,video)))
def mv_to_person(root):
  # import pdb;pdb.set_trace()
  in_path = join(root,'videos')
  for i in range(5):
    os.makedirs(join(root,'person'+str(i),'videos'), exist_ok=True)
  L = sorted(os.listdir(in_path))
  for pid in range(5):
    out = join(root,'person'+str(pid),'videos')
    for cam in range(16):
      name = L[cam*5+pid]
      cam_id = name.split('_')[0]
      cmd = 'mv {}/{} {}/{}.mp4'.format(in_path,name,out,cam_id)
      # print(cmd)
      run_cmd(cmd)
      

    


def extract_img(root):
  cmd = 'python3 apps/preprocess/extract_image.py {}'.format(root)
  run_cmd(cmd)

if __name__ == '__main__':

  # import argparse
  # parser = argparse.ArgumentParser()
  # parser.add_argument('split', type=str)
  # args = parser.parse_args()
  # root = '/nas/dataset/HalfCapture/'
  root = '/dellnas/dataset/HalfCapture/'
  orig_path = root+'capture0617'
  newfile = '220617-0/'
  out_path = root+'process0617' #+newfile
  if False:
    with Timer('mv'):
      mv_orig_data(orig_path, join(out_path, 'videos'))
  if True:
    with Timer('mv to person'):
      mv_to_person(out_path)
  if False:
    with Timer('extract_img'):
      extract_img(out_path)


  Timer.report()
    
