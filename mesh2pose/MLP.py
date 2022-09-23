import json
import os
from os.path import join
import trimesh
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# import imghdr
# from tkinter import Frame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from MLP_datasets import CompHand, FreiHAND



class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)

class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))
        
    def forward(self, x):
        return F.relu(x + self.fc_block(x))

class SMPLParamRegressor(nn.Module):

    def __init__(self):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        # self.layers = nn.Sequential(FCBlock(1723 * 6, 1024),
        #                             FCResBlock(1024, 1024),
        #                             FCResBlock(1024, 1024),
        #                             nn.Linear(1024, 24 * 3 * 3 + 10))

        self.layers = nn.Sequential(FCBlock(778 * 3, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 48))
        # self.use_cpu_svd = use_cpu_svd

    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        # breakpoint()
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)


        # rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        # betas = x[:, 24*3*3:].contiguous()
        # rotmat = rotmat.view(-1, 3, 3).contiguous()
        # orig_device = rotmat.device
        # if self.use_cpu_svd:
        #     rotmat = rotmat.cpu()
        # U, S, V = batch_svd(rotmat)

        # rotmat = torch.matmul(U, V.transpose(1,2))
        # det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        # with torch.no_grad():
        #     for i in range(rotmat.shape[0]):
        #         det[i] = torch.det(rotmat[i])
        # rotmat = rotmat * det
        # rotmat = rotmat.view(batch_size, 24, 3, 3)
        # rotmat = rotmat.to(orig_device)
        return x

def to_onnx():
    device = torch.device('cuda')

    MLP = SMPLParamRegressor().to(device)
    # MLP.load_state_dict(torch.load('/home/xxx/MLP/0905/mano_mesh2pose1_8.pt'))
    MLP.load_state_dict(torch.load('/home/xxx/MLP/0906/mano_mesh2pose1_11.pt'))
    MLP.eval()
    dummy_input = torch.randn(2,778,3).to(device)
    
    print('cvt to onnx')
    torch.onnx.export(MLP,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "/home/xxx/datasets/realtime_test_out/onnx/mesh2pose_11.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=16,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 

from handmodel2 import HandNet
import cv2
from manopth.manolayer import ManoLayer
mano_layer_pca = ManoLayer(mano_root='/home/xxx/work/EasyMocapPublic/data/bodymodels/manov1.2/', ncomps=24,use_pca=True,flat_hand_mean=False) 
def debug_vis(img_path, mesh, gt_pose):
    # breakpoint()
    # frame = cv2.imread(img_path)[:, ::-1, ::-1]
    frame = cv2.imread(img_path)
    from easymocap.visualize.pyrender_wrapper import plot_meshes

    pred = mesh
    vertex = pred.cpu().numpy()

    # vertex_mean = [vertex[:,0].mean(),vertex[:,1].mean(),vertex[:,2].mean()]
    for mean_id in range(3):
        vertex[:,mean_id] -= vertex[:,mean_id].mean() #vertex_mean[mean_id]
    # breakpoint()
    mesh_gt_r = {
        'id': 0,
        'vertices': vertex,
        'faces': mano_layer_pca.th_faces,
        'name': 'handr_{}'.format(0)
    }
    R = np.eye(3)
    K=[[frame.shape[0],0,frame.shape[0]/2],
    [0,frame.shape[1],frame.shape[1]/2],
    [0,0,1]]
    T=[0,0,0.4]
    T = np.array(T).reshape(3,1)
    K = np.array(K)
    out_img = plot_meshes(frame, {0: mesh_gt_r}, K, R, T,mode='hstack')

    from easymocap.socket.base_client import BaseSocketClient

    client = BaseSocketClient('127.0.0.1:9999')
    d_pose = [0]*111+gt_pose.reshape(48)[3:].tolist()
    data={'id':0, 'poses':[]}
    data["Th"]=np.array([0.0,-3,2], dtype=np.float32).reshape(1,3)
    data['expression']=np.array([0.0]*10, dtype=np.float32).reshape((1,10))
    data['type'] = 'smplh'
    data['poses'] = d_pose
    data['shapes'] = np.array([0.0]*16, dtype=np.float32).reshape((1,16))
    data['poses'] = np.array(data['poses'], dtype=np.float32).reshape((1,156))
    data['Rh']=np.array(data['poses'][0,:3], dtype=np.float32).reshape(1,3)


    client.send_any([data])
    cv2.imshow('vis',out_img)
    key = cv2.waitKey(0) & 0xFF
    # if i==0:
    #     # out_img = cv2.flip(out_img, 1)
    #     # cv2.imshow('handl', out_img)
    #     cv2.imwrite(join('/home/xxx/datasets/realtime_test_out/0809','handl','{:06d}.jpg'.format(self.img_id_L)), out_img)
    #     self.img_id_L+=1
    # else:
    #     # cv2.imshow('handr', out_img)
    #     cv2.imwrite(join('/home/xxx/datasets/realtime_test_out/0809','handr','{:06d}.jpg'.format(self.img_id_R)), out_img)
    #     self.img_id_R+=1




def Train():
    debug = False #True

    device = torch.device('cuda')
    hand_engine = '/home/xxx/datasets/realtime_test_out/onnx/handmesh_2hand_fp16.engine'
    mano_path = '/home/xxx/work/EasyMocapPublic/data/bodymodels/manov1.2/'
    regressor_path = "/home/xxx/work/EasyMocapPublic/data/smplx/"
    hand_net = HandNet(device)
    smpl_param_regressor = SMPLParamRegressor().to(device)
    optimizer = torch.optim.Adam(params=list(smpl_param_regressor.parameters()),
                                           lr=3e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
    criterion_regr = nn.MSELoss().to(device)

    from easymocap.annotator.file_utils import getFileList, save_json, save_annot

    root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'
    # filename_lists = getFileList(root,'.obj')
    # filename_lists=['base_pose/model_mano/2/3.obj']
    
    # objFilePath = join(root,filename_lists[0])
    # jsonfilepath = objFilePath.replace('model_mano','mano_pose').replace('.obj','.json')
    # obj = trimesh.load(objFilePath,process=False)
    # breakpoint()


    dataset_CompHand = CompHand(device)
    dataset_FreiHAND = FreiHAND(device)
    datasets_concat = ConcatDataset([dataset_CompHand, dataset_FreiHAND])
    # train_loader = DataLoader(dataset_CompHand, batch_size=32, shuffle= None, sampler=None)
    # train_loader = DataLoader(dataset_FreiHAND, batch_size=32, shuffle= None, sampler=None)
    train_loader = DataLoader(datasets_concat, batch_size=64, shuffle= None, sampler=None)

    # minloss = 99999999
    st_epoch = 9
    # smpl_param_regressor.load_state_dict(torch.load('/home/xxx/MLP/0905/mano_mesh2pose1_8.pt'))
    smpl_param_regressor.load_state_dict(torch.load('/home/xxx/MLP/0906/mano_mesh2pose1_11.pt'))

    smpl_param_regressor.train()
    for epoch in range(st_epoch,50):
        print("epoch{}".format(epoch))
        # for i in range(len(filename_lists)):
        for i, Input_ in tqdm(enumerate(train_loader)):
            # breakpoint()
            pred_mesh = hand_net(Input_['img'])
            pred_mesh[:,:,:] -= pred_mesh[:,0,:].reshape((pred_mesh.shape[0],1,3)).float()
            pred_pose = smpl_param_regressor(pred_mesh)
            gt_pose = Input_['poses'].to(device).float()
            loss_regr_pose = criterion_regr(pred_pose, gt_pose)

            if debug:
                # breakpoint()

                # from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, get_affine_transform
                # from MLP_datasets import base_transform
                # for j in [1004, 1258, 18, 360, 499,  56, 704, 866, 1031, 1266,  213, 372, 520, 597, 741, 884, 1052, 1267,  278, 406, 538, 621, 768, 912, 1110, 1277,   30, 420, 539, 659, 793, 939, 1180,  153,  330, 466, 563, 663, 83,]:
                #     img = cv2.imread('/home/xxx/work/otherwork/Hand_detect/MeshTransformer/samples/data/{}.jpg'.format(j))
                #     bbox = [0,0,img.shape[1],img.shape[0],1]
                #     center, scale = box_to_center_scale(bbox, 128, 128,scale_factor=0.8)#1.23
                #     trans = get_affine_transform(center, scale, rot=0, output_size=(128,128))
                #     img = cv2.warpAffine(
                #         np.array(img), trans,
                #         (128,128),
                #         flags=cv2.INTER_LINEAR)
                #     # img = img[:, ::-1, ::-1]
                #     img = cv2.resize(img, (128,128))
                #     img = torch.from_numpy(base_transform(img, size=128)).to(device)
                #     pred_mesh = hand_net(img[None])
                    
                #     debug_vis('/home/xxx/work/otherwork/Hand_detect/MeshTransformer/samples/data/{}.jpg'.format(j), pred_mesh[0], gt_pose[0])
                # breakpoint()

                for j in range(Input_['img'].shape[0]):
                    debug_vis(Input_['path'][j], pred_mesh[j], gt_pose[j])
                breakpoint()

            # input_ = np.array(obj.vertices)
            # input_ -= input_[0,:]
            # input_[:, 0] *= -1
            # input_*=0.2
            # input_ = torch.tensor(input_[None]).to(device)
            # input_ = input_.float()
            # ii = torch.zeros((2,input_.shape[1],input_.shape[2])).to(device)
            # ii[0] = input_[0]
            # ii[1] = input_[0]
            # # breakpoint()
            # pred_pose = smpl_param_regressor(ii)#.cpu()
            
            # breakpoint()
            # with open(jsonfilepath,'r') as f:
            #     data = json.load(f)
            # gt_pose = torch.tensor(data['poses']).to(device).reshape((1,-1))
            # gt_Rh = torch.tensor(data['Rh']).to(device).reshape((1,-1))
            # gt_pose = torch.cat((gt_Rh,gt_pose),1)
            # loss_regr_pose = criterion_regr(pred_pose[0], gt_pose[0])


            

            optimizer.zero_grad()
            loss_regr_pose.backward()
            optimizer.step()
        # if epoch % 1==0:

        torch.save(smpl_param_regressor.state_dict(), '/home/xxx/MLP/0906/mano_mesh2pose1_{}.pt'.format(epoch))
            # torch.save(smpl_param_regressor.state_dict(), '/home/users/wangjunhao/mano_mesh2pose_{}.pt'.format(epoch))
    # smpl_param_regressor.sa
    breakpoint()
    
    torch.save(smpl_param_regressor.state_dict(), '/home/xxx/MLP/mano_mesh2pose.pt')
    # torch.save(smpl_param_regressor.state_dict(), '/home/users/wangjunhao/mano_mesh2pose.pt')

    smpl_param_regressor.load_state_dict(torch.load('/home/xxx/MLP/mano_mesh2pose.pt'))

    breakpoint()

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--onnx', action='store_true')
    # args = parser.parse_args()
    # if args.onnx:
    #     to_onnx()
    # else:
    to_onnx()
    # Train()





"""
python scripts/my/MLP.py \
    --phase 'demo' \
    --exp_name 'mobrecon' \
    --dataset 'FreiHAND' \
    --model 'mobrecon' \
    --backbone 'DenseStack' \
    --device_idx -1 \
    --size 128 \
    --out_channels 32 64 128 256 \
    --seq_length 9 9 9 9 \
    --resume 'mobrecon_densestack_dsconv.pt'

"""