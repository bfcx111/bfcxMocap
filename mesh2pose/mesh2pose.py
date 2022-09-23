import os
from os.path import join
import numpy as np
from tqdm import tqdm

import numpy as np
import torch

from manopth.manolayer import ManoLayer
from easymocap.annotator.file_utils import getFileList, save_json, save_annot

debug = True

dataset_name='FreiHAND'
dataset_name='process_Compdata'

if dataset_name=='FreiHAND':
    hand_type = 'right'
else:
    hand_type = 'left'


print('{}__{}'.format(dataset_name,hand_type))
# hand_type = 'left'
# root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'
# root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/real_world_testset/real_hand_3D_mesh'
# breakpoint()
easymocap_path = os.path.abspath(os.path.join(os.getcwd()))
easymocap_path = os.path.abspath(join(easymocap_path, '..', '..'))
mano_path = easymocap_path+'/data/bodymodels/manov1.2/'
regressor_path = easymocap_path+"/data/smplx/"
# file_dir = easymocap_path+'/library/pysmpl/scripts/mesh.npz'

# def Load():
#     data = np.load(file_dir)
#     mesh = data['mesh']

#     from easymocap.bodymodel.smpl import load_regressor
#     joint_regressor = load_regressor(join(regressor_path,"J_regressor_mano_RIGHT.txt"))
if hand_type=='left':
    mano_model_path=mano_path+'/MANO_LEFT.pkl'
    regressor_path_=join(regressor_path,"J_regressor_mano_LEFT.txt")
else:
    mano_model_path=mano_path+'/MANO_RIGHT.pkl'
    regressor_path_=join(regressor_path,"J_regressor_mano_RIGHT.txt")
class FitMano():
    def __init__(self,mano_model_path, regressor_path_) -> None:
        from pysmpl import MANO
        self.fittermodel = MANO(
            # 模型路径在EasyMocap下面，注意文件地址
            model_path=mano_model_path, #mano_path+'/MANO_RIGHT.pkl',
            model_type='mano',
            device='cpu',
            regressor_path= regressor_path_, #join(regressor_path,"J_regressor_mano_RIGHT.txt"),
            use_pose_blending=True,
            use_shapre_blending=True,
            num_pca_comps=24,
            use_pca=True,
            use_flat_mean=False,
            gender='male'
        )
        self.fittermodel.weight_shape['debug'] = 0
        self.fittermodel.weight_pose['debug'] = 0
        if hand_type=='left':
            mano_layer = ManoLayer(mano_root=mano_path, side='left', flat_hand_mean=False, use_pca=False)
            self.mano_layer_pca = ManoLayer(mano_root=mano_path, side='left', ncomps=24,use_pca=True,flat_hand_mean=False) 
        else:
            mano_layer = ManoLayer(mano_root=mano_path, flat_hand_mean=False, use_pca=False) # load right hand MANO model

            self.mano_layer_pca = ManoLayer(mano_root=mano_path, ncomps=24,use_pca=True,flat_hand_mean=False) 
        self.joint_regressor = mano_layer.th_J_regressor.numpy()
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673] # mesh vertex idx (right hand)
        thumbtip_onehot = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        indextip_onehot = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        middletip_onehot = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        ringtip_onehot = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        pinkytip_onehot = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]

    def __call__(self, mesh):
        res_pose=[]
        # breakpoint()
        mesh = torch.from_numpy(mesh).cuda()[None]
        mesh = mesh.float()
        joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(mesh.shape[0], 1, 1),mesh)
        for i in range(len(joint_img_from_mesh)):
            pose = self.joint2pose(joint_img_from_mesh[i].cpu().numpy())
            res_pose.append(pose)
        return res_pose


    def joint2pose(self, keypoints):
        
        # keypoints = joint_img_from_mesh[i].cpu().numpy()
        keypoints /= self.fittermodel.check_scale(keypoints)
        keypoints = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        params = self.fittermodel.init_params(1)
        params_shape = self.fittermodel.fit3DShape(keypoints[None], params)
        params_RT = self.fittermodel.init3DRT(keypoints, params_shape)
        params_poses = self.fittermodel.fit3DPose(keypoints[None], params_RT)


        # th_pose_coeffs = params_poses['poses']
        # th_pose_coeffs = torch.from_numpy(th_pose_coeffs)
        # th_hand_pose_coeffs = th_pose_coeffs[:, self.mano_layer_pca.rot:self.mano_layer_pca.rot +
        #                                         self.mano_layer_pca.ncomps]
        # pose = th_hand_pose_coeffs.mm(self.mano_layer_pca.th_selected_comps)
        # pose = self.mano_layer_pca.th_hands_mean + pose
        # pose = pose.numpy()
        # params_poses['poses']=pose

        return params_poses

        

import scipy.io as scio


def process_Ge():
    
    root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/real_world_testset/'
    fitmano = FitMano(mano_model_path,regressor_path_)
    filename_lists = getFileList(root,'.obj')

    dataFile = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/real_world_testset/pose_gt.mat'
    ge_data = scio.loadmat(dataFile)
    ge_data = ge_data['pose_gt']
    # breakpoint()
    # breakpoint()
    for i in tqdm(range(ge_data.shape[0])):

        joint = ge_data[i] / 100

        objFilePath = join(root,filename_lists[i])
        print(objFilePath)

        # with open(objFilePath) as file:
        #     mesh = []
        #     breakpoint()
        #     while True:
        #         line = file.readline()
        #         if not line:
        #             break
        #         strs = line.split(" ")
        #         if strs[0] == "v":
        #             mesh.append((float(strs[1]), float(strs[2]), float(strs[3])))
        #         if strs[0] == "f":
        #             break
        # points原本为列表，需要转变为矩阵，方便处理          
        # mesh = np.array(mesh)

        params_poses = fitmano.joint2pose(joint)
        pose = params_poses['poses']
        # breakpoint()
        if hand_type=='left':
            pose = pose.reshape((-1,3))
            pose[:, 1::3] = -pose[:, 1::3]
            pose[:, 2::3] = -pose[:, 2::3]
            params_poses['poses'] = pose.reshape((1,45))
            Rh = params_poses['Rh']
            Rh[:, 1::3] = -Rh[:, 1::3]
            Rh[:, 2::3] = -Rh[:, 2::3]
            params_poses['Rh'] = Rh

        pose = params_poses['poses']
        # breakpoint()
        if debug:

            from easymocap.socket.base_client import BaseSocketClient

            client = BaseSocketClient('127.0.0.1:9999')
            d_pose = [0]*111+pose.reshape(45).tolist()
            data={'id':0, 'poses':[]}
            data["Th"]=np.array([0.0,-3,2], dtype=np.float32).reshape(1,3)
            data['expression']=np.array([0.0]*10, dtype=np.float32).reshape((1,10))
            data['type'] = 'smplh'
            data['poses'] = d_pose
            data['shapes'] = np.array([0.0]*16, dtype=np.float32).reshape((1,16))
            data['poses'] = np.array(data['poses'], dtype=np.float32).reshape((1,156))
            data['Rh']=np.array(data['poses'][0,:3], dtype=np.float32).reshape(1,3)


            client.send_any([data])

        for key in params_poses.keys():
            params_poses[key] = params_poses[key].tolist()
        # outfilename = objFilePath.replace('model_mano','mano_pose').replace('.obj','.json')
        outfilename = objFilePath.replace('real_hand_3D_mesh','mano_pose').replace('.obj','.json')

        save_annot(outfilename,params_poses)



def process_Compdata():
    #hand_type = 'left'
    root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'
    
    fitmano = FitMano(mano_model_path,regressor_path_)
    #Compdata
    filename_lists = getFileList(root,'.obj')
    import trimesh
    # filename_lists=['base_pose/model_mano/2/3.obj']
    #right hand
    for i in tqdm(range(len(filename_lists))):
        objFilePath = join(root,filename_lists[i])
        print(objFilePath)

        obj = trimesh.load(objFilePath,process=False)

        mesh = np.array(obj.vertices)
        # with open(objFilePath) as file:
        #     mesh = []
        #     breakpoint()
        #     while True:
        #         line = file.readline()
        #         if not line:
        #             break
        #         strs = line.split(" ")
        #         if strs[0] == "v":
        #             mesh.append((float(strs[1]), float(strs[2]), float(strs[3])))
        #         if strs[0] == "f":
        #             break
        # points原本为列表，需要转变为矩阵，方便处理          
        # mesh = np.array(mesh)

        params_poses = fitmano(mesh)[0]
        pose = params_poses['poses']
        # breakpoint()
        if hand_type=='left':
            pose = pose.reshape((-1,3))
            pose[:, 1::3] = -pose[:, 1::3]
            pose[:, 2::3] = -pose[:, 2::3]
            params_poses['poses'] = pose.reshape((1,45))
            Rh = params_poses['Rh']
            Rh[:, 1::3] = -Rh[:, 1::3]
            Rh[:, 2::3] = -Rh[:, 2::3]
            params_poses['Rh'] = Rh

        pose = params_poses['poses']
        # breakpoint()
        if debug:

            from easymocap.socket.base_client import BaseSocketClient

            client = BaseSocketClient('127.0.0.1:9999')
            d_pose = [0]*111+pose.reshape(45).tolist()
            data={'id':0, 'poses':[]}
            data["Th"]=np.array([0.0,-3,2], dtype=np.float32).reshape(1,3)
            data['expression']=np.array([0.0]*10, dtype=np.float32).reshape((1,10))
            data['type'] = 'smplh'
            data['poses'] = d_pose
            data['shapes'] = np.array([0.0]*16, dtype=np.float32).reshape((1,16))
            data['poses'] = np.array(data['poses'], dtype=np.float32).reshape((1,156))
            data['Rh']=np.array(data['poses'][0,:3], dtype=np.float32).reshape(1,3)


            client.send_any([data])

        for key in params_poses.keys():
            params_poses[key] = params_poses[key].tolist()
        outfilename = objFilePath.replace('model_mano','mano_pose').replace('.obj','.json')
        # outfilename = objFilePath.replace('real_hand_3D_mesh','mano_pose').replace('.obj','.json')

        save_annot(outfilename,params_poses)



def process_FreiHAND():
    #hand_type = 'right'
    # root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'
    root = '/home/xxx/DGPU/dellnas/users/wangjunhao/data/FreiHAND/training/mesh/'

    fitmano = FitMano(mano_model_path,regressor_path_)
    #Compdata
    filename_lists = getFileList(root,'.ply')
    import trimesh
    # filename_lists=['base_pose/model_mano/2/3.obj']
    #right hand
    for i in tqdm(range(len(filename_lists))):
        objFilePath = join(root,filename_lists[i])
        print(objFilePath)

        obj = trimesh.load(objFilePath,process=False)

        mesh = np.array(obj.vertices)
        # with open(objFilePath) as file:
        #     mesh = []
        #     breakpoint()
        #     while True:
        #         line = file.readline()
        #         if not line:
        #             break
        #         strs = line.split(" ")
        #         if strs[0] == "v":
        #             mesh.append((float(strs[1]), float(strs[2]), float(strs[3])))
        #         if strs[0] == "f":
        #             break
        # points原本为列表，需要转变为矩阵，方便处理          
        # mesh = np.array(mesh)

        params_poses = fitmano(mesh)[0]
        pose = params_poses['poses']
        # breakpoint()
        # if hand_type=='left':
        #     pose = pose.reshape((-1,3))
        #     pose[:, 1::3] = -pose[:, 1::3]
        #     pose[:, 2::3] = -pose[:, 2::3]
        #     params_poses['poses'] = pose.reshape((1,45))
        #     Rh = params_poses['Rh']
        #     Rh[:, 1::3] = -Rh[:, 1::3]
        #     Rh[:, 2::3] = -Rh[:, 2::3]
        #     params_poses['Rh'] = Rh

        pose = params_poses['poses']
        # breakpoint()
        if debug:

            from easymocap.socket.base_client import BaseSocketClient

            client = BaseSocketClient('127.0.0.1:9999')
            d_pose = [0]*111+pose.reshape(45).tolist()
            data={'id':0, 'poses':[]}
            data["Th"]=np.array([0.0,-3,2], dtype=np.float32).reshape(1,3)
            data['expression']=np.array([0.0]*10, dtype=np.float32).reshape((1,10))
            data['type'] = 'smplh'
            data['poses'] = d_pose
            data['shapes'] = np.array([0.0]*16, dtype=np.float32).reshape((1,16))
            data['poses'] = np.array(data['poses'], dtype=np.float32).reshape((1,156))
            data['Rh']=np.array(data['poses'][0,:3], dtype=np.float32).reshape(1,3)


            client.send_any([data])

        for key in params_poses.keys():
            params_poses[key] = params_poses[key].tolist()
        # outfilename = objFilePath.replace('model_mano','mano_pose').replace('.obj','.json')
        outfilename = objFilePath.replace('mesh','mano_pose').replace('.ply','.json')

        save_annot(outfilename,params_poses)
        # break


# process_Ge()
# process_FreiHAND()
process_Compdata()