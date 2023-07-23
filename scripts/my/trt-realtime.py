'''
  Author: wjh
'''
# 这个脚本用于单个USB相机半身动作捕捉
# 例如：
# 1. yolo
# 2. body -- PARE
# 3. hand -- handmesh
import torch
# import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import json
from multiprocessing import Process, Queue
from copy import deepcopy

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import os.path as osp

from easymocap.socket.base_client import BaseSocketClient
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, get_affine_transform
from easymocap.multistage.torchgeometry import axis_angle_to_euler, euler_to_axis_angle
from easymocap.annotator.file_utils import save_json

from my_realtime_need_file import get_single_image_crop_demo, letterbox, non_max_suppression, scale_coords, do_inference_v2, allocate_buffers, base_transform

from os.path import join
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import sys
# from scipy.spatial.transform import Rotation as R
import scipy
from manopth.manolayer import ManoLayer






class Yolo_trt:
    def __init__(self, engine_path, device) -> None:
        from collections import OrderedDict, namedtuple
        w = engine_path
        import tensorrt as trt  
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.context = model.create_execution_context()
        self.batch_size = bindings['images'].shape[0]

        self.bindings = bindings


        self.img_size = [640,640]
        self.stride = 32

        self.device = device
        self.min_detect_thres = 0.3
        self.conf_thres=0.25
        self.iou_thres=0.45
        self.max_det=1000
        self.classes=None
        self.agnostic_nms=False
        self.lab_id={
            0: 'person',
            1: 'handl',
            2: 'handr',
            3: 'face'
        }

        self.smbox = Smooth_Box()


    @torch.no_grad()
    def __call__(self, frame):
        torch.cuda.synchronize()
        t1 = time.time()

        img = letterbox(frame, self.img_size, stride=self.stride, auto=False)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device)

        # im = im.half() #if model.fp16 else im.float()  # uint8 to fp16/32
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        torch.cuda.synchronize()
        t2 = time.time()
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        pred = self.bindings['output'].data
        torch.cuda.synchronize()
        t3 = time.time()

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        arrays = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                for idx in range(det.shape[0]):
                    arrays.append(det[idx,:].tolist() + [self.lab_id[int(det[idx,-1])]])
        
        torch.cuda.synchronize()
        t4 = time.time()

        arrays = np.array(arrays)
        results = {}
        for i, res in enumerate(arrays):
            name = res[6] + str(i)
            bbox = res[:5].astype(np.float)
            # bbox = rescalebox(bbox)
            if bbox[4] < self.min_detect_thres:
                continue
            if name not in results.keys():
                results[name] = change_box(bbox, frame.shape) #bbox #
            else:
                if results[name][-1] < bbox[-1]:
                    results[name] = change_box(bbox, frame.shape) #bbox #
        res={}
        for key in results.keys():
            if key[:-1] not in res.keys():
                res[key[:-1]] = results[key]
            else:
                if res[key[:-1]][-1] < results[key][-1]:
                    res[key[:-1]] = results[key]

        results = res
        for key in ['person', 'handl', 'handr']:
            if key not in results.keys():
                results[key] = [0,0,100,100,0]
        results['person'] = [0,0,frame.shape[1],frame.shape[0],results['person'][-1]]
        # results['person'] = [0,0,frame.shape[1],frame.shape[0],1]
        results = self.smbox(results)

        torch.cuda.synchronize()
        t5 = time.time()

        print('yolo timet21',t2-t1)
        print('yolo timet32',t3-t2)
        print('yolo timet43',t4-t3)
        print('yolo timet54',t5-t4)
        return results
        
    
class BodyNet:
    def __init__(self, engine_path, device) -> None: 
        # breakpoint()
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        # engine_path = 'D:/Work/real-time-bodyandhand/onnx/pare_fp16.trt'
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

    @torch.no_grad()
    def __call__(self, frame, bbox):
        torch.cuda.synchronize()
        t1 = time.time()
        if bbox[-1]<=0:
            return {'pred_pose':[0.0]*66, 'pred_shape':np.array([0.0]*10).reshape((1,10)), 'time':0,'pred_cam':[0,0,0]}
        
        img = frame.copy()
        dets = np.array(xyxy2ccwh(bbox)).reshape((1,4))
        # orig_height, orig_width = img.shape[:2]        
        bbox = dets[0]
        torch.cuda.synchronize()
        t2 = time.time()
        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                img,
                bbox,
                kp_2d=None ,
                scale=1.0,
                crop_size=224
        )

        self.inputs[0].host = norm_img.float().detach().numpy()#.to(self.device)
        torch.cuda.synchronize()
        t3 = time.time()
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        torch.cuda.synchronize()
        t4 = time.time()
        output = {
            'pred_pose': np.array(trt_outputs[3]).reshape((1,24,3,3)), 
            'pred_shape': np.array(trt_outputs[2]).reshape((1,10))
        }

        poses = []
        for i in range(len(output['pred_pose'][0])):
            res,j = cv2.Rodrigues(output['pred_pose'][0][i])
            poses = poses+res.reshape(3).tolist()
        output['pred_pose'] = poses[:66]
        torch.cuda.synchronize()
        t5 = time.time()
        print('body timet21',t2-t1)
        print('body timet32',t3-t2)
        print('body timet43',t4-t3)
        print('body timet54',t5-t4)
        return output 


class HandNet:
    def __init__(self, engine_path, mano_path, regressor_path, device) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        # from manopth.manolayer import ManoLayer
        # mano_layer = ManoLayer(mano_root=mano_path, flat_hand_mean=False, use_pca=False) # load right hand MANO model
        self.mano_layer_pca = ManoLayer(mano_root=mano_path, ncomps=24,use_pca=True,flat_hand_mean=False) 
        # self.joint_regressor = mano_layer.th_J_regressor.numpy()
        # self.fingertip_vertex_idx = [745, 317, 444, 556, 673] # mesh vertex idx (right hand)
        # thumbtip_onehot = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        # indextip_onehot = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        # middletip_onehot = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        # ringtip_onehot = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        # pinkytip_onehot = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        # self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        # self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]

        from easymocap.bodymodel.smpl import load_regressor
        self.joint_regressor = np.array(load_regressor(join(regressor_path,"J_regressor_mano_RIGHT.txt")))
        self.size = 128

        self.device = device



        from library.pysmpl.pysmpl.smplcpp import MANO
        self.fittermodel = MANO(
            # 模型路径在EasyMocap下面，注意文件地址
            model_path=mano_path+'/MANO_RIGHT.pkl',
            model_type='mano',
            device='cpu',
            regressor_path= join(regressor_path,"J_regressor_mano_RIGHT.txt"),
            use_pose_blending=True,
            use_shapre_blending=True,
            num_pca_comps=24,
            use_pca=True,
            use_flat_mean=False,
            gender='male'
        )
        self.fittermodel.weight_shape['debug'] = 0
        self.fittermodel.weight_pose['debug'] = 0
        

    @torch.no_grad()
    def __call__(self, frame, bboxes):        
        torch.cuda.synchronize()
        t1 = time.time()
        batch_size = sum([bbox['bbox'][-1]>0 for bbox in bboxes])
        if batch_size ==0:
            return {'poses':[0.0]*90,'Rh':[[0,0,0],[0,0,0]],'Rh_flag':[False,False]}
        images_tensor = torch.zeros((2, 3, 128, 128), device=self.device)  # (height, width)
        count = 0
        use=[]
        for ann in bboxes:
            bbox = ann['bbox']
            flip = ann['fliplr']
            if bbox[-1]==0:
                continue
            use.append(flip)
            center, scale = box_to_center_scale(bbox, self.size, self.size,scale_factor=1.23)
            trans = get_affine_transform(center, scale, rot=0, output_size=(self.size,self.size))
            img = cv2.warpAffine(
                frame, trans,
                (int(self.size), int(self.size)),
                flags=cv2.INTER_LINEAR)
            if flip:
                img = cv2.flip(img, 1)
            img = img[..., ::-1]
            img = cv2.resize(img, (self.size, self.size))
            input = torch.from_numpy(base_transform(img, size=self.size)).unsqueeze(0).to(self.device)
            images_tensor[count] = input#.half()
            count += 1

        torch.cuda.synchronize()
        t2 = time.time()

        self.inputs[0].host = images_tensor.detach().cpu().numpy()#.to(self.device)  .float()
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # out = {'mesh_pred': torch.tensor(trt_outputs[1]).reshape((2,778,3)).to(self.device)}
        out = {'mesh_pred': np.array(trt_outputs[1]).reshape((2,778,3))}
        mesh_coord_img = out['mesh_pred'] * 0.2
        torch.cuda.synchronize()
        t3 = time.time()
        # joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(mesh_coord_img.shape[0], 1, 1),mesh_coord_img)
        # breakpoint()
        joint_img_from_mesh = np.matmul(self.joint_regressor, mesh_coord_img)
        out_pose=[]
        Rh_list=[]
        Rh_flag = []

        for i in range(len(use)):#joint_img_from_mesh.shape[0]
            keypoints = joint_img_from_mesh[i]#.cpu().numpy()
            keypoints /= self.fittermodel.check_scale(keypoints)
            keypoints = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

            params = self.fittermodel.init_params(1)
            params_shape = self.fittermodel.fit3DShape(keypoints[None], params)
            params_RT = self.fittermodel.init3DRT(keypoints, params_shape)

            params_poses = self.fittermodel.fit3DPose(keypoints[None], params_RT)

            th_pose_coeffs = params_poses['poses']

            # th_pose_coeffs = torch.from_numpy(th_pose_coeffs)
            th_hand_pose_coeffs = th_pose_coeffs[:, self.mano_layer_pca.rot:self.mano_layer_pca.rot +
                                                 self.mano_layer_pca.ncomps]
            # PCA components --> axis angles
            pose = np.matmul(th_hand_pose_coeffs,self.mano_layer_pca.th_selected_comps)
            # pose = th_hand_pose_coeffs.mm(self.mano_layer_pca.th_selected_comps)
            pose = self.mano_layer_pca.th_hands_mean + pose
            # pose = pose.numpy()
            if use[i]:
                pose = pose.reshape((15,3))
                pose[:, 1::3] = -pose[:, 1::3]
                pose[:, 2::3] = -pose[:, 2::3]

                Rh = params_poses['Rh']
                Rh[:, 1::3] = -Rh[:, 1::3]
                Rh[:, 2::3] = -Rh[:, 2::3]
                params_poses['Rh'] = Rh
            pose = pose.reshape(45)
            out_pose = out_pose+pose.tolist()
            Rh_list.append(params_poses['Rh'].tolist())
            Rh_flag.append(True)
    
        if len(use)==1:
            if use[0]:
                Rh_flag = [True, False]
                Rh_list.append([0,0,0])
                out_pose = out_pose + [0.0]*45
            else:
                Rh_flag =  [False, True]
                Rh_list = [[0,0,0], Rh_list[0]]
                out_pose = [0.0]*45 + out_pose
        torch.cuda.synchronize()
        t4 = time.time()
        print('hand timet21',t2-t1)
        print('hand timet32',t3-t2)
        print('hand timet43',t4-t3)
        return {'poses':out_pose,'Rh':Rh_list,'Rh_flag':Rh_flag}




def xyxy2ccwh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    c = [(bbox[2] + bbox[0])/2 , (bbox[3] + bbox[1])/2]
    w=max(w,h)
    h=w
    return [c[0],c[1],w,h]



def rescalebox(box,rescale=1.4,shape=(9999,9999,3)):
    c = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    w = (box[2]-box[0]) * rescale
    h = (box[3]-box[1]) * rescale
    w = max(w,h)
    h = w
    bbox=[
        max(0,c[0]-w/2),
        max(0,c[1]-h/2),
        min(shape[1],c[0]+w/2),
        min(shape[0],c[1]+h/2),
        box[4]
        ]
    return bbox

def change_box(bbox,shape):
    lmt_c = 80
    lmt_b =5
    c = [(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
    if (bbox[0]<lmt_b and abs(c[0])<lmt_c) or (bbox[1]<lmt_b and abs(c[1])<lmt_c) or ((shape[1]-bbox[2])<lmt_b and abs(shape[1]-c[0])<lmt_c) or ((shape[0]-bbox[3])<lmt_b and abs(shape[0]-c[1])<lmt_c):
        return [0,0,100,100,0]
    return bbox

class Smooth_Data:
    def __init__(self) -> None:
        self.size ={
            'Rh':25,
            'body':9,
            'handl':5,
            'handr':5
        }
        self.smdata={
            'Rh':[],
            'body':[],
            'handl':[],
            'handr':[]
        }
    def __call__(self,data):
        self.smdata['Rh'].append(data['Rh'])
        self.smdata['body'].append(data['poses'][0,:66])
        self.smdata['handl'].append(data['poses'][0,66:66+45])
        self.smdata['handr'].append(data['poses'][0,66+45:])
        res={'Rh':[],
            'body':[],
            'handl':[],
            'handr':[]}
        for key in self.smdata:
            if len(self.smdata[key])>self.size[key]:
                self.smdata[key].pop(0)
            res[key] = (sum(self.smdata[key])/len(self.smdata[key])).reshape(1,self.smdata[key][0].shape[-1])
        data['Rh'] = res['Rh']
        data['poses'] = np.concatenate((res['body'],res['handl'],res['handr']),axis=1)#?
        return data 
class Smooth_Box:
    def __init__(self) -> None:
        self.size ={
            'handl':3,
            'handr':3
        }
        self.smdata={
            'handl':[],
            'handr':[]
        }
    def __call__(self,data):
        for key in ['handl','handr']:
            if data[key][-1]>0:
                self.smdata[key].append(np.array(data[key]))
            elif len(self.smdata[key])>0 :
                self.smdata[key].pop(0)

            if len(self.smdata[key])>self.size[key]:#0 and self.num[key]>
                self.smdata[key].pop(0)
            if len(self.smdata[key])>0:
                data[key] = (sum(self.smdata[key])/len(self.smdata[key])).tolist()

        return data 

def get_R(poses, cfg, st):
    res = st.copy()
    for i in cfg:
        res = res @ cv2.Rodrigues(poses[i,:])[0]
    return  res
def process_poses_mano(poses, hand_Rh, flag):
    if sum(flag) == 0:
        return poses 
    
    poses = poses.reshape((-1,3))
    cfg={'rt': [0,3,6,9],
        'r': [14,17,19],
        'l': [13,16,18]
    }
    RA = get_R(poses, cfg['rt'],np.eye(3))

    if flag[0] :
        RL = get_R(poses, cfg['l'],RA)
        tmppose = np.matrix(RL).I @ cv2.Rodrigues(np.array(hand_Rh[0]))[0]
        tmppose = cv2.Rodrigues(tmppose)[0]
        poses[20,:] = tmppose.reshape(3)

        e20 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[20,:]).reshape(-1,3))
        e20 = e20.as_euler('ZYX', degrees=True)
        e18 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[18,:]).reshape(-1,3))
        e18 = e18.as_euler('ZYX', degrees=True)
        e20[0,2] =  e20[0,2]/2
        e18[0,2] += e20[0,2]
        e20 = scipy.spatial.transform.Rotation.from_euler('ZYX', e20, degrees=True)
        e20 = e20.as_rotvec()
        e18 = scipy.spatial.transform.Rotation.from_euler('ZYX', e18, degrees=True)
        e18 = e18.as_rotvec()
        poses[20,:] = e20
        poses[18,:] = e18
    if flag[1] : #and sum(np.array(hand_Rh[1])!=0)>0:
        RR = get_R(poses, cfg['r'],RA)
        tmppose = np.matrix(RR).I @ cv2.Rodrigues(np.array(hand_Rh[1]))[0]
        tmppose = cv2.Rodrigues(tmppose)[0]
        poses[21,:] = tmppose.reshape(3)
        
        e21 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[21,:]).reshape(-1,3))
        e21 = e21.as_euler('ZYX', degrees=True)
        e19 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(poses[19,:]).reshape(-1,3))
        e19 = e19.as_euler('ZYX', degrees=True)
        e21[0,2] =  e21[0,2]/2
        e19[0,2] += e21[0,2]
        e21 = scipy.spatial.transform.Rotation.from_euler('ZYX', e21, degrees=True)
        e21 = e21.as_rotvec()
        e19 = scipy.spatial.transform.Rotation.from_euler('ZYX', e19, degrees=True)
        e19 = e19.as_rotvec()
        poses[21,:] = e21
        poses[19,:] = e19

    return poses.reshape((1,-1))


class IPUSB:  
    def __init__(self) -> None:
        self.run = True
        self.camnames = list([0])
        self.maxsize = 1
        self.queue = Queue(maxsize=self.maxsize*2) # keep 0.1s
        self.queue_flag = True #Queue(maxsize=2)
        self.thread = Process(target=self.start_camera_thread, args=(self.queue, self.queue_flag))
        self.thread.start()

    def start_camera_thread(self, queue, queue_flag):
        cap = cv2.VideoCapture('https://10.0.235.143:8080/video')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        w, h = 1280, 720
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 60)
        for i in range(10):
            cap.read()
        while queue_flag:#.qsize() == 0:
            while queue.qsize() > self.maxsize:
                queue.get()
            flag, image = cap.read()
            if not flag:
                for _it in range(10):
                    time.sleep(0.01)
                    flag, image = cap.read()
                    if flag:break
                else:
                    break
            stamp = int(time.time() * 1000)
            queue.put({
                'sub': 0,
                'timestamp': stamp,
                'image': image,
            })
        cap.release()

    def capture(self):
        results = {}
        camnames = self.camnames
        flag = False
        while not flag and self.run:
            data = self.queue.get()
            sub = data['sub']
            results[sub] = data
            timestamp = [val['timestamp'] for val in results.values()]
            flag1 = max(timestamp) - min(timestamp) < 40 # less than 1/fps/2 ms
            flag2 = len(timestamp) == len(camnames)
            flag = flag1 and flag2

        timestamp = [results[cam]['timestamp'] - results[camnames[0]]['timestamp'] for cam in camnames]
        return True, results[0]['image']
    
    def release(self):
        self.run = False
        self.queue_flag = False

    def __del__(self):
        self.release()

def process_poses(data, output_body, output_hand):
    data['poses'] = output_body['pred_pose']+output_hand['poses']
    data['shapes'] = output_body['pred_shape'].tolist()[0]
    data['shapes'] = np.array(data['shapes']+[0.0]*6, dtype=np.float32).reshape((1,16))
    data['poses'] = np.array(data['poses'], dtype=np.float32).reshape((1,156))
    data['Rh']=np.array(data['poses'][0,:3], dtype=np.float32).reshape(1,3)
    data['poses'] = process_poses_mano(data['poses'], output_hand['Rh'], output_hand['Rh_flag'])


    data['Rh'][0,0] = 3.2
    data['poses'][0,:3] = 0
    data['poses'][0,3:9] = 0
    data['poses'][0,12:18] = 0
    data['poses'][0,21:27] = 0.0
    data['poses'][0,30:36] = 0.0
    data['poses'][0,9:12] = 0

    # #sit
    # data['poses'][0,3] = -1.45
    # data['poses'][0,6] = -1.45
    # data['poses'][0,12] = 1.6
    # data['poses'][0,15] = 1.6

    #stand
    data['poses'][0,3] = 0
    data['poses'][0,6] = 0
    data['poses'][0,12] = 0
    data['poses'][0,15] = 0


    e15 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(data['poses'][0,45:48]).reshape(-1,3))
    e15 = e15.as_euler('XYZ', degrees=True)
    e12 = scipy.spatial.transform.Rotation.from_rotvec(torch.from_numpy(data['poses'][0,36:39]).reshape(-1,3))
    e12 = e12.as_euler('XYZ', degrees=True)
    e15[0,0] = 0
    e12[0,0] = 0
    e15[0,2] = 0
    e12[0,2] = 0
    lmt = 15
    if e15[0,1]>lmt:
        e15[0,1]=lmt
    elif e15[0,1]<-lmt:
        e15[0,1]=-lmt
    if e12[0,1]>lmt:
        e12[0,1]=lmt
    elif e12[0,1]<-lmt:
        e12[0,1]=-lmt

    e15 = scipy.spatial.transform.Rotation.from_euler('XYZ', e15, degrees=True)
    e15 = e15.as_rotvec()
    e12 = scipy.spatial.transform.Rotation.from_euler('XYZ', e12, degrees=True)
    e12 = e12.as_rotvec()
    data['poses'][0,36:39] = e12
    data['poses'][0,45:48] = e15

    return data



def main(yolo_engine, body_engine, hand_engine, mano_path, regressor_path):
    # breakpoint()
    device = torch.device('cuda')
    print('yolo')
    yolo_trt = Yolo_trt(yolo_engine, device)
    print('body')
    body_net = BodyNet(body_engine, device)
    print('hand')
    hand_net = HandNet(hand_engine, mano_path, regressor_path, device)
    data={'id':0, 'poses':[]}
    # data["Th"]=np.array([0.0,-3,2], dtype=np.float32).reshape(1,3)
    data["Th"]=np.array([0.0,0,0], dtype=np.float32).reshape(1,3)
    data['expression']=np.array([0.0]*10, dtype=np.float32).reshape((1,10))
    data['type'] = 'smplh'
    smooth = Smooth_Data()

    sum_All_time = 0
    sum_read_time = 0
    sum_yolo_time = 0
    sum_body_time = 0
    sum_hand_time = 0
    sum_num = 0

    # breakpoint()
    if args.input_type == 'ip-camera':
        cap_thread = IPUSB()
        def get_image():
            return cap_thread.capture()
    elif args.input_type == 'usb-camera':
        cap = cv2.VideoCapture(-1)
        # cap = cv2.VideoCapture('https://10.0.235.143:8080/video')
        w, h = 1280, 720
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 60)
        def get_image():
            return cap.read()
    elif args.input_type == 'folder':
        count_id = 0
        from os.path import join
        import os
        root = '/dellnas/dataset/DeskStage/0706/data/images/0'
        # root = '/home/xxx/datasets/test-yolo-media/0810/images'
        # root = 'D:/Work/real-time-bodyandhand/onnx/'
        def get_image():
            nonlocal count_id
            imgname = join(root, '{:06d}.jpg'.format(count_id))
            if not os.path.exists(imgname):
                return False, None
            count_id += 1
            return True, cv2.imread(imgname)

    while True:

        torch.cuda.synchronize()
        all_st = time.time()

        flag, frame = get_image()
        if flag ==False:
            break
        vis = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        torch.cuda.synchronize()
        yolo_st = time.time()
        # results = yolo_trt(frame)
        results = yolo_trt(vis)
        torch.cuda.synchronize()
        body_st = time.time()
        output_body = body_net(frame, results['person'])
        torch.cuda.synchronize()
        hand_st = time.time()
        
        # results['handl'] = rescalebox(results['handl'],rescale=1.2,shape=frame.shape)
        # results['handr'] = rescalebox(results['handr'],rescale=1.2,shape=frame.shape)

        results['handl'] = change_box(results['handl'], frame.shape)
        results['handr'] = change_box(results['handr'], frame.shape)
        
        output_hand = hand_net(vis,[{'bbox': results['handl'], 'fliplr': True}, {'bbox': results['handr'], 'fliplr': False}])

        torch.cuda.synchronize()
        hand_end = time.time()

        data = process_poses(data, output_body, output_hand)
        
        data = smooth(data)

        torch.cuda.synchronize()
        all_end = time.time()
        if args.sent in ['local', 'all']:
            client.send_any([data])
        
        if args.sent in ['unity', 'all']:
            d2 = deepcopy(data)
            d2['type'] = 'smpl'
            d2["Rh"]=np.array([0.0,0.0,0.0], dtype=np.float32).reshape(1,3)
            d2["Th"]=np.array([0.0,0.0,0.0], dtype=np.float32).reshape(1,3)
            sent_results = {'annots':[d2]}
            client_unity.to_euler(sent_results)
            client_unity.send_str(sent_results)

        # if flag:
        #     for name, bbox in results.items():
        #         plot_bbox(vis, bbox, name)
        #     vis = cv2.resize(vis,(1280,720))
        #     cv2.imshow('vis', vis)
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == ord('q'):
        #         break
        #     elif k == ord('p'):
        #         print(count_id)

        print('All time:',all_end - all_st)
        print('read time:',yolo_st - all_st)
        print('yolo time:',body_st - yolo_st)
        print('body time:',hand_st - body_st)
        print('hand time:',all_end - hand_st)

        sum_All_time  += all_end - all_st
        sum_read_time += yolo_st - all_st
        sum_yolo_time += body_st - yolo_st
        sum_body_time += hand_st - body_st
        sum_hand_time += all_end - hand_st
        sum_num += 1
        if sum_num>=500:
            break
    print('All  time:',sum_All_time/sum_num)
    print('read time:',sum_read_time/sum_num)
    print('yolo time:',sum_yolo_time/sum_num)
    print('body time:',sum_body_time/sum_num)
    print('hand time:',sum_hand_time/sum_num)
    print('hand time:',sum_num)
    cv2.destroyAllWindows()


    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', default='usb-camera', choices=['folder', 'usb-camera', 'ip-camera'])
    parser.add_argument('--sent', default='not-use', choices=['not-use', 'local', 'unity', 'all'])
    parser.add_argument('--two_hand_interaction', action='store_true')
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # easymocap_path = 'D:/Work/easymocappublic'
    engine_root = '/dellnas/users/wangjunhao/data/onnx'
    yolo_engine_path = engine_root+'/yolo_fp16.engine'
    # body_engine_path = engine_root+'/pare_fp16.engine'
    hand_engine_path = engine_root+'/handmesh_2hand_fp16.engine'
    body_engine_path = engine_root+'/pare_fp16_s.engine'
    # body_engine_path = engine_root+'/pare_fp16_old.engine'
    # hand_engine_path = 'D:/Work/real-time-bodyandhand/onnx/handmesh_2hand_fp16.engine'
    # yolo_engine_path = 'D:/Work/real-time-bodyandhand/onnx/yolo_fp16_s.engine'
    # body_engine_path = 'D:/Work/real-time-bodyandhand/onnx/pare_fp16.trt'

    # mano_path = easymocap_path+'/data/bodymodels/manov1.2/'
    # regressor_path = easymocap_path+"/data/smplx/"
    # breakpoint()

    # yolo_engine_path = '/home/xxx/share/yolo-0810-best.engine'
    # body_engine_path = '/home/xxx/datasets/realtime_test_out/onnx/pare_test0728-1-fp16.trt'
    # hand_engine_path = '/home/xxx/datasets/realtime_test_out/onnx/handmesh_2hand_fp16.engine'
    # two_hand_engine = '/home/xxx/intaghand_0810_fp16.engine'
    # body_engine_path = '/home/xxx/datasets/realtime_test_out/onnx/pare_0812_fp16.engine'
    mano_path = './data/bodymodels/manov1.2/'
    regressor_path = "./data/smplx/"

    if args.sent in ['local', 'all']:
        client = BaseSocketClient('127.0.0.1:9999')
        # client = BaseSocketClient('127.0.0.1',9999)
    
    if args.sent in ['unity', 'all']:
        client_unity = BaseSocketClient('10.0.1.1:12340:associator_output')
    main(yolo_engine=yolo_engine_path, body_engine=body_engine_path, hand_engine=hand_engine_path, mano_path=mano_path, regressor_path=regressor_path)

'''
$env:HTTP_PROXY="http://10.0.0.11:7890"
conda activate mocap
cd D:\Work\real-time-bodyandhand\onnx
python trt-realtime.py --sent local --input_type usb-camera
python trt-realtime.py --input_type usb-camera

cd D:\Work\easymocappublic
python apps/vis/vis_server_all.py --opts hostport 0.0.0.0:9999 
'''
