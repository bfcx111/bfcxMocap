'''
  @ Author: wjh
'''

from os.path import join
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
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

class ManoLayer(Module):
    def __init__(self, manoPath, center_idx=9, use_pca=True, new_skel=False):
        super(ManoLayer, self).__init__()

        self.center_idx = center_idx
        self.use_pca = use_pca
        self.new_skel = new_skel

        manoData = pickle.load(open(manoPath, 'rb'), encoding='latin1')

        self.new_order = [0,
                          13, 14, 15, 16,
                          1, 2, 3, 17,
                          4, 5, 6, 18,
                          10, 11, 12, 19,
                          7, 8, 9, 20]

        # 45 * 45: PCA mat
        self.register_buffer('hands_components', torch.from_numpy(manoData['hands_components'].astype(np.float32)))
        hands_components_inv = torch.inverse(self.hands_components)
        self.register_buffer('hands_components_inv', hands_components_inv)
        # 16 * 778, J_regressor is a scipy csc matrix
        J_regressor = manoData['J_regressor'].tocoo(copy=False)
        location = []
        data = []
        for i in range(J_regressor.data.shape[0]):
            location.append([J_regressor.row[i], J_regressor.col[i]])
            data.append(J_regressor.data[i])
        i = torch.LongTensor(location)
        v = torch.FloatTensor(data)
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i.t(), v, torch.Size([16, 778])).to_dense(),
                             persistent=False)
        # 16 * 3
        self.register_buffer('J_zero', torch.from_numpy(manoData['J'].astype(np.float32)), persistent=False)
        # 778 * 16
        self.register_buffer('weights', torch.from_numpy(manoData['weights'].astype(np.float32)), persistent=False)
        # (778, 3, 135)
        self.register_buffer('posedirs', torch.from_numpy(manoData['posedirs'].astype(np.float32)), persistent=False)
        # (778, 3)
        self.register_buffer('v_template', torch.from_numpy(manoData['v_template'].astype(np.float32)), persistent=False)
        # (778, 3, 10) shapedirs is <class 'chumpy.reordering.Select'>
        if isinstance(manoData['shapedirs'], np.ndarray):
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs']).float(), persistent=False)
        else:
            self.register_buffer('shapedirs', torch.Tensor(manoData['shapedirs'].r.copy()).float(), persistent=False)
        # 45
        self.register_buffer('hands_mean', torch.from_numpy(manoData['hands_mean'].astype(np.float32)), persistent=False)

        self.faces = manoData['f']  # 1538 * 3: faces

        self.parent = [-1, ]
        for i in range(1, 16):
            self.parent.append(manoData['kintree_table'][0, i])

    def get_faces(self):
        return self.faces

    def train(self, mode=True):
        self.is_train = mode

    def eval(self):
        self.train(False)

    def pca2axis(self, pca):
        rotation_axis = pca.mm(self.hands_components[:pca.shape[1]])  # bs * 45
        rotation_axis = rotation_axis + self.hands_mean
        return rotation_axis  # bs * 45

    def pca2Rmat(self, pca):
        return self.axis2Rmat(self.pca2axis(pca))

    def axis2Rmat(self, axis):
        # axis: bs x 45
        rotation_mat = rodrigues_batch(axis.view(-1, 3))
        rotation_mat = rotation_mat.view(-1, 15, 3, 3)
        return rotation_mat

    def axis2pca(self, axis):
        # axis: bs x 45
        pca = axis - self.hands_mean
        pca = pca.mm(self.hands_components_inv)
        return pca

    def Rmat2pca(self, R):
        # R: bs x 15 x 3 x 3
        return self.axis2pca(self.Rmat2axis(R))

    def Rmat2axis(self, R):
        # R: bs x 3 x 3
        R = R.view(-1, 3, 3)
        temp = (R - R.permute(0, 2, 1)) / 2
        L = temp[:, [2, 0, 1], [1, 2, 0]]  # bs x 3
        sin = torch.norm(L, dim=1, keepdim=False)  # bs
        L = L / (sin.unsqueeze(-1) + 1e-8)

        temp = (R + R.permute(0, 2, 1)) / 2
        temp = temp - torch.eye((3), dtype=R.dtype, device=R.device)
        temp2 = torch.matmul(L.unsqueeze(-1), L.unsqueeze(1))
        temp2 = temp2 - torch.eye((3), dtype=R.dtype, device=R.device)
        temp = temp[:, 0, 0] + temp[:, 1, 1] + temp[:, 2, 2]
        temp2 = temp2[:, 0, 0] + temp2[:, 1, 1] + temp2[:, 2, 2]
        cos = 1 - temp / (temp2 + 1e-8)  # bs

        sin = torch.clamp(sin, min=-1 + 1e-7, max=1 - 1e-7)
        theta = torch.asin(sin)

        # prevent in-place operation
        theta2 = torch.zeros_like(theta)
        theta2[:] = theta
        idx1 = (cos < 0) & (sin > 0)
        idx2 = (cos < 0) & (sin < 0)
        theta2[idx1] = 3.14159 - theta[idx1]
        theta2[idx2] = -3.14159 - theta[idx2]
        axis = theta2.unsqueeze(-1) * L

        return axis.view(-1, 45)

    def get_local_frame(self, shape):
        # output: frame[..., [0,1,2]] = [splay, bend, twist]
        # get local joint frame at zero pose
        with torch.no_grad():
            shapeBlendShape = torch.matmul(self.shapedirs, shape.permute(1, 0)).permute(2, 0, 1)
            v_shaped = self.v_template + shapeBlendShape  # bs * 778 * 3
            j_tpose = torch.matmul(self.J_regressor, v_shaped)  # bs * 16 * 3
            j_tpose_21 = torch.cat((j_tpose, v_shaped[:, [744, 320, 444, 555, 672]]), axis=1)
            j_tpose_21 = j_tpose_21[:, self.new_order]
            frame = build_mano_frame(j_tpose_21)
        return frame  # bs x 15 x 3 x 3

    @staticmethod
    def buildSE3_batch(R, t):
        # R: bs * 3 * 3
        # t: bs * 3 * 1
        # return: bs * 4 * 4
        bs = R.shape[0]
        pad = torch.zeros((bs, 1, 4), dtype=R.dtype, device=R.device)
        pad[:, 0, 3] = 1.0
        temp = torch.cat([R, t], 2)  # bs * 3 * 4
        return torch.cat([temp, pad], 1)

    @staticmethod
    def SE3_apply(SE3, v):
        # SE3: bs * 4 * 4
        # v: bs * 3
        # return: bs * 3
        bs = v.shape[0]
        pad = torch.ones((bs, 1), dtype=v.dtype, device=v.device)
        temp = torch.cat([v, pad], 1).unsqueeze(2)  # bs * 4 * 1
        return SE3.bmm(temp)[:, :3, 0]

    def forward(self, root_rotation, pose, shape, trans=None, scale=None):
        # input
        # root_rotation : bs * 3 * 3
        # pose : bs * ncomps or bs * 15 * 3 * 3
        # shape : bs * 10
        # trans : bs * 3 or None
        # scale : bs or None
        bs = root_rotation.shape[0]

        if self.use_pca:
            rotation_mat = self.pca2Rmat(pose)
        else:
            rotation_mat = pose

        shapeBlendShape = torch.matmul(self.shapedirs, shape.permute(1, 0)).permute(2, 0, 1)
        v_shaped = self.v_template + shapeBlendShape  # bs * 778 * 3

        j_tpose = torch.matmul(self.J_regressor, v_shaped)  # bs * 16 * 3

        Imat = torch.eye(3, dtype=rotation_mat.dtype, device=rotation_mat.device).repeat(bs, 15, 1, 1)
        pose_shape = rotation_mat.view(bs, -1) - Imat.view(bs, -1)  # bs * 135
        poseBlendShape = torch.matmul(self.posedirs, pose_shape.permute(1, 0)).permute(2, 0, 1)
        v_tpose = v_shaped + poseBlendShape  # bs * 778 * 3

        SE3_j = []
        R = root_rotation
        t = (torch.eye(3, dtype=pose.dtype, device=pose.device).repeat(bs, 1, 1) - R).bmm(j_tpose[:, 0].unsqueeze(2))
        SE3_j.append(self.buildSE3_batch(R, t))
        for i in range(1, 16):
            R = rotation_mat[:, i - 1]
            t = (torch.eye(3, dtype=pose.dtype, device=pose.device).repeat(bs, 1, 1) - R).bmm(j_tpose[:, i].unsqueeze(2))
            SE3_local = self.buildSE3_batch(R, t)
            SE3_j.append(torch.matmul(SE3_j[self.parent[i]], SE3_local))
        SE3_j = torch.stack(SE3_j, dim=1)  # bs * 16 * 4 * 4

        j_withoutTips = []
        j_withoutTips.append(j_tpose[:, 0])
        for i in range(1, 16):
            j_withoutTips.append(self.SE3_apply(SE3_j[:, self.parent[i]], j_tpose[:, i]))

        # there is no boardcast matmul for sparse matrix for now (pytorch 1.6.0)
        SE3_v = torch.matmul(self.weights, SE3_j.view(bs, 16, 16)).view(bs, -1, 4, 4)  # bs * 778 * 4 * 4

        v_output = SE3_v[:, :, :3, :3].matmul(v_tpose.unsqueeze(3)) + SE3_v[:, :, :3, 3:4]
        v_output = v_output[:, :, :, 0]  # bs * 778 * 3

        jList = j_withoutTips + [v_output[:, 745], v_output[:, 317], v_output[:, 444], v_output[:, 556], v_output[:, 673]]

        j_output = torch.stack(jList, dim=1)
        j_output = j_output[:, self.new_order]

        if self.center_idx is not None:
            center = j_output[:, self.center_idx:(self.center_idx + 1)]
            v_output = v_output - center
            j_output = j_output - center

        if scale is not None:
            scale = scale.unsqueeze(1).unsqueeze(2)  # bs * 1 * 1
            v_output = v_output * scale
            j_output = j_output * scale

        if trans is not None:
            trans = trans.unsqueeze(1)  # bs * 1 * 3
            v_output = v_output + trans
            j_output = j_output + trans

        if self.new_skel:
            j_output[:, 5] = (v_output[:, 63] + v_output[:, 144]) / 2
            j_output[:, 9] = (v_output[:, 271] + v_output[:, 220]) / 2
            j_output[:, 13] = (v_output[:, 148] + v_output[:, 290]) / 2
            j_output[:, 17] = (v_output[:, 770] + v_output[:, 83]) / 2

        return v_output, j_output

def rodrigues_batch(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)
def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

def pre_Interhand_mano(out, split):
    

    
    mano_path = {'left': os.path.join('/nas/users/wangjunhao/otherwork/IntagHand/misc/mano', 'MANO_LEFT.pkl'),
                 'right': os.path.join('/nas/users/wangjunhao/otherwork/IntagHand/misc/mano', 'MANO_RIGHT.pkl')}
    # mano_path=get_mano_path()
    mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
    fix_shape(mano_layer)



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
        mano_params = json.load(f)
    caps = os.listdir(root_path+'images/'+split)
    
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

                        

                        # if(sum(ptl[:,2]==0)+sum(ptr[:,2]==0)>32):
                        #     continue

                        # 
                        

                        K = np.eye(3)
                        K[0, 0] = focal[0]
                        K[1, 1] = focal[1]
                        K[0, 2] = princpt[0]
                        K[1, 2] = princpt[1]
                        RT = np.eye(4)
                        R = np.array(camrot)
                        T = np.array(campos)/1000
                        T = -np.dot(R, T.reshape(3,1)).reshape(3) # -Rt -> t

                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        img_height, img_width, _ = img.shape

                        # prev_depth = None

                        ann ={"personID": 0, "left":{}, "right":{}}

                        # mano_dict={}
                        for hand_type in ["left", "right"]:
                            mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
                            mano_pose, mano_shape, mano_trans = [], [], []
                            hand_pose = []
                            if mano_param is not None:
                                if True:
                                    # mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
                                    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
                                    root_pose = mano_pose[0].view(1, 3)
                                    hand_pose = mano_pose[1:, :].view(1, -1)
                                    # hand_pose = hand_pose.view(1, -1, 3)
                                    mano = mano_layer[hand_type]
                                    mean_pose = mano.hands_mean
                                    hand_pose = mano.axis2pca(hand_pose + mean_pose)
                                    shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                                    trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
                                    root_pose = rodrigues_batch(root_pose)

                                    handV, handJ = mano_layer[hand_type](root_pose, hand_pose, shape, trans=trans)
                                    params = {'R': root_pose.numpy(), 'pose': hand_pose.numpy(), 'shape': shape.numpy(), 'trans': trans.numpy()}


                                    # params = data['mano_params'][hand_type]
                                    handV, handJ = mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                                        torch.from_numpy(params['pose']).float(),
                                                                        torch.from_numpy(params['shape']).float(),
                                                                        trans=torch.from_numpy(params['trans']).float())

                                    handV = handV[0].numpy()
                                    handJ = handJ[0].numpy()
                                    handV = handV @ R.T + T
                                    handJ = handJ @ R.T + T

                                    handV2d = handV @ K.T
                                    handV2d = handV2d[:, :2] / handV2d[:, 2:]
                                    handJ2d = handJ @ K.T
                                    handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

                                    # import pdb;pdb.set_trace()
                                    plt.figure()
                                    plt.imshow(img)
                                    plt.plot(handV2d[:,0],handV2d[:,1], '.', color='red')
                                    plt.draw()
                                    plt.savefig('/nas/users/wangjunhao/out/test/{}_{}.jpg'.format(frame_idx,hand_type))
                                    plt.tight_layout()
                                    plt.close()
                                    # cv2.imwrite('/nas/users/wangjunhao/out/test/{}_{}.jpg'.format(frame_idx,hand_type),img)

                                    

                                mano_pose = np.array(mano_param['pose']).reshape(-1, 3).tolist()
                                mano_shape = mano_param['shape']
                                mano_trans = mano_param['trans']
                                
                                

                            ann[hand_type]={
                                "pose" : mano_pose,
                                "shape" : mano_shape,
                                "trans" : mano_trans,
                                # "hand_pose" : hand_pose
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
                            pass
                            # rendered_img = render_mesh(img, mesh, mano_layer[hand_type].faces, {'focal': focal, 'princpt': princpt})
                            
                            # cv2.imwrite('render_original_img.jpg', rendered_img)
                            # print("************1****************")
                            
                            # cv2.imshow('vis',img)
                            # cv2.imshow('vis2',images)
                            # k = cv2.waitKey(0) & 0xFF
                            # if k == ord('q'):
                            #     break
                        if False:
                            out_path_img = join(out,split,'images', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.jpg'.format(frame_idx))
                            if not os.path.exists(os.path.dirname(out_path_img)):
                                os.makedirs(os.path.dirname(out_path_img))
                            shutil.copy(img_path,out_path_img)
                            save_json(join(out, split, 'annots', str(capture_id)+'+'+str(seq_name)+'+'+str(cam)+'+{}.json'.format(frame_idx)), res_write)

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