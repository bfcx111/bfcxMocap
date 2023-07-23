'''
  @ Author: wjh
'''

from os.path import join
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

# from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
# from easymocap.estimator.wrapper_base import bbox_from_keypoints
# from easymocap.mytools.file_utils import read_json, save_json
import cv2
import pdb
from tqdm import tqdm
import shutil
from torch.nn import Module
import pickle
import smplx
import trimesh
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import torch



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

def check_kpts(pt, shape, d=5):
    x=pt[:,0]
    y=pt[:,1]
    v=pt[:,2]  
    # pdb.set_trace()
    ff = (((0-d)<x) & (x<(shape[1]+d)) & ((0-d)<y) &(y<(shape[0]+d)))^0
    pt[:,2]=v * ff


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


# def render_mesh(img, mesh, face, cam_param):
#     # mesh
#     mesh = trimesh.Trimesh(mesh, face)
#     rot = trimesh.transformations.rotation_matrix(
# 	np.radians(180), [1, 0, 0])
#     mesh.apply_transform(rot)
#     material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
#     mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
#     scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
#     scene.add(mesh, 'mesh')
    
#     focal, princpt = cam_param['focal'], cam_param['princpt']
#     camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
#     scene.add(camera)
 
#     # renderer
#     renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
#     # light
#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
#     light_pose = np.eye(4)
#     light_pose[:3, 3] = np.array([0, -1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([0, 1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([1, 1, 2])
#     scene.add(light, pose=light_pose)

#     # render
#     rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#     rgb = rgb[:,:,:3].astype(np.float32)
#     valid_mask = (depth > 0)[:,:,None]

#     # save to image
#     img = rgb * valid_mask + img * (1-valid_mask)
#     return img

def pre_Interhand_mano(out, split):
    

    
    # mano layer
    smplx_path = '/nas/users/wangjunhao/data/human_model_files/'
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}

    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1


    # mano_path = {'left': os.path.join('/nas/users/wangjunhao/otherwork/IntagHand/misc/mano', 'MANO_LEFT.pkl'),
    #              'right': os.path.join('/nas/users/wangjunhao/otherwork/IntagHand/misc/mano', 'MANO_RIGHT.pkl')}
    # # mano_path=get_mano_path()
    # mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
    #                        'left': ManoLayer(mano_path['left'], center_idx=None)}
    # fix_shape(mano_layer)

    root_path = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/'
    # pdb.set_trace()
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'camera')),'r') as f:
        cameras = json.load(f)

    # joint_num = 21
    # root_joint_idx = {'right': 20, 'left': 41}
    # joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}

    # with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'joint_3d')),'r') as f:
    #     joints = json.load(f)

    # with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_{}.json'.format(split,'data')),'r') as f:
    #     data = json.load(f)
    with open(join(root_path, 'annotations', split,'InterHand2.6M_{}_MANO_NeuralAnnot.json'.format(split)),'r') as f:
        mano = json.load(f)
    caps = os.listdir(root_path+'images/'+split)
    id_mesh=0
    for cap in caps:
        seqs = os.listdir(join(root_path+'images/'+split,cap))
        for seq in tqdm(seqs):
            cams = os.listdir(join(root_path+'images/'+split,cap,seq))
            for cam_name in cams:
                cam = cam_name.split('cam')[1]
                if cam[1] == '1':
                    continue
                frames = os.listdir(join(root_path+'images/'+split,cap,seq,cam_name))
                frames.sort()
                for i in range(len(frames)):
                    if i%5==0:
                        capture_id = cap.split('Capture')[1]
                        seq_name = seq
                        # pdb.set_trace()
                        frame_idx = int(frames[i].split('.')[0].split('image')[1])
                        # frame_idx = frames[i].split('.')[0].split('image')[1]
        
                        img_path = join(root_path, 'images', split, cap, seq, cam_name, frames[i])
                        images = cv2.imread(img_path)
                        if images.mean() < 10:
                            continue
                        
                        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)

                        out_path_img = join(out,split,'images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx))
                        if not os.path.exists(os.path.dirname(out_path_img)):
                            os.makedirs(os.path.dirname(out_path_img))

                        # if(sum(ptl[:,2]==0)+sum(ptr[:,2]==0)>32):
                        #     continue

                        # shutil.copy(img_path,out_path_img)
                        

                        K = np.eye(3)
                        K[0, 0] = focal[0]
                        K[1, 1] = focal[1]
                        K[0, 2] = princpt[0]
                        K[1, 2] = princpt[1]
                        RT = np.eye(4)
                        R = np.array(camrot)
                        #可视化失败因为/1000...
                        # ?存的时候要除1000
                        T = np.array(campos) /1000
                        T = -np.dot(R, T.reshape(3,1)) # -Rt -> t

                        img = cv2.imread(img_path)
                        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        img_height, img_width, _ = img.shape

                        prev_depth = None

                        ann ={"personID": 0, "left":{}, "right":{}}
                        for hand_type in ["left", "right"]:
                            mano_param = mano[str(capture_id)][str(frame_idx)][hand_type]
                            mano_pose, mano_shape, mano_trans = [], [], []
                            hand_pose = []
                            if mano_param is not None:
                                if True:
                                    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
                                    root_pose = mano_pose[0].view(1,3)
                                    hand_pose = mano_pose[1:,:].view(1,-1)
                                    shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
                                    trans = torch.FloatTensor(mano_param['trans']).view(1,3)
                                    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                                    mesh = output.vertices[0].numpy() * 1000 # meter to milimeter

                                    # mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + T.reshape(1,3)
                                    t, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
                                    t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
                                    mesh = np.dot(R, mesh.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
                                    # save_obj(mesh, mano_layer[hand_type].faces, join('/nas/users/wangjunhao/out/test/test_mesh/mesh/' + hand_type +'_'+str(id_mesh)+ '.obj'))
                                    # shutil.copy(img_path,join('/nas/users/wangjunhao/out/test/test_mesh/img/', '+{}.jpg'.format(id_mesh)))
                                    # id_mesh+=1
                                    mesh = mesh / 1000 # milimeter to meter
                                    mesh = trimesh.Trimesh(mesh, mano_layer[hand_type].faces)
                                    rot = trimesh.transformations.rotation_matrix(
                                        np.radians(180), [1, 0, 0])
                                    mesh.apply_transform(rot)
                                    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
                                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
                                    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
                                    scene.add(mesh, 'mesh')

                                    # add camera intrinsics
                                    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
                                    scene.add(camera)
                                    
                                    # renderer
                                    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=1.0)
                                
                                    # light
                                    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
                                    light_pose = np.eye(4)
                                    light_pose[:3, 3] = np.array([0, -1, 1])
                                    scene.add(light, pose=light_pose)
                                    light_pose[:3, 3] = np.array([0, 1, 1])
                                    scene.add(light, pose=light_pose)
                                    light_pose[:3, 3] = np.array([1, 1, 2])
                                    scene.add(light, pose=light_pose)

                                    # render
                                    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                                    rgb = rgb[:,:,:3].astype(np.float32)
                                    depth = depth[:,:,None]
                                    valid_mask = (depth > 0)
                                    if prev_depth is None:
                                        render_mask = valid_mask
                                        img = rgb * render_mask + img * (1 - render_mask)
                                        prev_depth = depth
                                    else:
                                        render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth==0)
                                        img = rgb * render_mask + img * (1 - render_mask)
                                        prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

                                mano_pose = np.array(mano_param['pose']).reshape(-1, 3).tolist()
                                mano_shape = mano_param['shape']
                                mano_trans = mano_param['trans']
                                
                                

                            ann[hand_type]={
                                "pose" : mano_pose,
                                "shape" : mano_shape,
                                "trans" : mano_trans,
                                "hand_pose" : hand_pose
                            }
                        
                        res_write={
                            "filename" : join('images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx)),
                            "height" : images.shape[0],
                            "width" : images.shape[1],
                            'K' : K, 
                            # 'dist' : np.zeros((1, 5)), 
                            'R' : R, 
                            'T' : T,
                            "annots" : [ann]
                        }

                        # coord = np.array(annot['world_coord'])/1000
                        # conf = np.array(annot['joint_valid'])
                        # coord[conf[:, 0]<1] = 0.
                        # k3d = np.hstack([coord, conf])
                        
                        # res = {
                        #     'id': 0,
                        #     'handl3d': k3d[:21],
                        #     'handr3d': k3d[21:]
                        # }
                        if True:
                            # rendered_img = render_mesh(img, mesh, mano_layer[hand_type].faces, {'focal': focal, 'princpt': princpt})
                            
                            # cv2.imwrite('render_original_img.jpg', rendered_img)
                            # print("************1****************")
                            img_res = np.concatenate((img,images),axis=1)

                            cv2.imwrite('/nas/users/wangjunhao/out/test/render/render_{}.jpg'.format(frame_idx),img_res)
                            pass
                            # cv2.imshow('vis',img)
                            # cv2.imshow('vis2',images)
                            # k = cv2.waitKey(0) & 0xFF
                            # if k == ord('q'):
                            #     break
                        if False:
                            pass
                            # save_json(join(out, split, 'annots', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.json'.format(frame_idx)), res_write)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--mano', action='store_true')
    # parser.add_argument('--host', type=str, default='127.0.0.1')
    # parser.add_argument('--port', type=int, default=9999)
    args = parser.parse_args()
    print(args.mano)
    # need = get_need(args.split)
    # pre_Interhand(args.out, args.split)
    if args.mano:
        pre_Interhand_mano(args.out, args.split)
    cv2.destroyAllWindows()
#python scripts/my/pre_Interhand.py /home/wangjunhao/data/Interhand2.6M --split='train'