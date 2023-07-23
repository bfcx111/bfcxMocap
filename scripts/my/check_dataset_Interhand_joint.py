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
import cv2
import pdb
def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


hand_edges = np.array([[0,1],[1,2],[2,3],[3,4],
                [0,5],[5,6],[6,7],[7,8],
                [0,9],[9,10],[10,11],[11,12],
                [0,13],[13,14],[14,15],[15,16],
                [0,17],[17,18],[18,19],[19,20]])

def run_cmd(cmd, verbo=True, bg=False):
    print('[run] ' + cmd, 'run')
    os.system(cmd)
    return []


def main():
     
    root_path = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/'
    # pdb.set_trace()
    split='val'
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'camera')),'r') as f:
        cameras = json.load(f)

    joint_num = 21
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}

    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'joint_3d')),'r') as f:
        joints = json.load(f)

    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'data')),'r') as f:
        data = json.load(f)


        
    info_ann = data['annotations']
    info_img = data['images']

    for i in range(len(info_img)):
        # if i <397000:
        #     continue
        # if i%1000!=0:
        #     continue
        ann = info_ann[i]
        img = info_img[i]


        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']
        img_path = join(root_path, 'images', split, img['file_name'])
        
        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)



        joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
        joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

        joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(joint_num*2)
        # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
        # joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
        # joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]

        # pt2d = np.zeros((21,3))
        # # pdb.set_trace()
        # pt2d[0,:2] = joint_img[20,:]
        # pt2d[0,2] = joint_valid[20]

        idx = [3,2,1,0]
        idx = np.array(idx)
        # for n in range(5):
        #     pt2d[1+n*4:5+n*4,:2] = joint_img[idx+n*4,:]
        #     pt2d[1+n*4:5+n*4,2] = joint_valid[idx+n*4]


        #left
        pt = np.zeros((21,3))
        pt[0,:2] = joint_img[41,:]
        pt[0,2] = joint_valid[41]
        for n in range(5):
            pt[1+n*4:5+n*4,:2] = joint_img[21+idx+n*4,:]
            pt[1+n*4:5+n*4,2] = joint_valid[21+idx+n*4]
        images = cv2.imread(img_path)
        # plot_keypoints_auto(images, pt2d, 4, lw=1)
        plot_keypoints_auto(images, pt, 4, lw=1)

        cv2.imshow('vis',images)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        # cv2.imwrite('/nas/users/wangjunhao/out/interhand/{}/{}.jpg'.format(split,i),images)
        # if i>500:
        #     break

    # out_path = '/nas/users/wangjunhao/out/interhand/{}/'.format(split)
    # cmd = 'ffmpeg -r 25 -i '+out_path+'handtest/'+args.test_epoch+'/%d.jpg -vcodec libx264 -r 25 '+out_path+'handtest/hand.mp4'
    # run_cmd(cmd)



if __name__ == "__main__":
    main()
