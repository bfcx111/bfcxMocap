'''
  Author: wjh
'''

import torch
import cv2
import numpy as np
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from easymocap.socket.base_client import BaseSocketClient
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto

import sys
sys.path.insert(0, '../otherwork/')

class Base:
    def __init__(self, root_path, model_name)->None:
        self.num=0
        self.time=0.0
        self.model_name = model_name
        self.root_path = root_path
    def get_fps(self):
        if self.num==0:
            return -1
        return self.num/self.time
    def get_numandtime(self):
        return self.num, self.time
    def report(self):
        print("{}:  num:{}, time:{}, fps:{}".format(self.model_name, self.num, self.time, self.get_fps()))
class BodyNet(Base):
    def __init__(self, ckpt, device, root_path, model_name='body') -> None:
        super(BodyNet, self).__init__(root_path, model_name)
        from PARE.pare.models import PARE
        from PARE.pare.core.config import update_hparams
        from PARE.pare.utils.train_utils import load_pretrained_model
        
        self.device = device
        model_cfg = update_hparams(self.root_path + '/otherwork/PARE/data/pare/checkpoints/pare_w_3dpw_config.yaml')
        self.model = PARE(
                    backbone=model_cfg.PARE.BACKBONE,
                    num_joints=model_cfg.PARE.NUM_JOINTS,
                    softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
                    num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
                    focal_length=model_cfg.DATASET.FOCAL_LENGTH,
                    img_res=model_cfg.DATASET.IMG_RES,
                    pretrained=model_cfg.TRAINING.PRETRAINED,
                    iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
                    num_iterations=model_cfg.PARE.NUM_ITERATIONS,
                    iter_residual=model_cfg.PARE.ITER_RESIDUAL,
                    shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
                    pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
                    pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
                    shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
                    pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
                    shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
                    use_keypoint_features_for_smpl_regression=model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
                    use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
                    use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
                    use_postconv_keypoint_attention=model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
                    use_scale_keypoint_attention=model_cfg.PARE.USE_SCALE_KEYPOINT_ATTENTION,
                    keypoint_attention_act=model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
                    use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
                    use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
                    use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
                    use_coattention=model_cfg.PARE.USE_COATTENTION,
                    num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
                    coattention_conv=model_cfg.PARE.COATTENTION_CONV,
                    use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
                    deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
                    use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
                    num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
                    branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
                    num_deconv_layers=model_cfg.PARE.NUM_DECONV_LAYERS,
                    num_deconv_filters=model_cfg.PARE.NUM_DECONV_FILTERS,
                    use_resnet_conv_hrnet=model_cfg.PARE.USE_RESNET_CONV_HRNET,
                    use_position_encodings=model_cfg.PARE.USE_POS_ENC,
                    use_mean_camshape=model_cfg.PARE.USE_MEAN_CAMSHAPE,
                    use_mean_pose=model_cfg.PARE.USE_MEAN_POSE,
                    init_xavier=model_cfg.PARE.INIT_XAVIER,
                ).to(device)
        self.ckpt_ = torch.load(ckpt)['state_dict']
        load_pretrained_model(self.model, self.ckpt_, overwrite_shape_mismatch=True, remove_lightning=True)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, frame, bbox):
        if bbox[-1]<=0:
            return {'pred_pose':[0.0]*66, 'pred_shape':np.array([0.0]*10).reshape((1,10)), 'time':0}
        
        from PARE.pare.utils.vibe_image_utils import get_single_image_crop_demo
        from PARE.pare.utils.demo_utils import (
            convert_crop_cam_to_orig_img,
            convert_crop_coords_to_orig_img,
            # prepare_rendering_results,
        )
        # from PARE.pare.utils.vibe_renderer import Renderer
        import colorsys
        img = frame.copy()
        dets = np.array(xyxy2ccwh(bbox)).reshape((1,4))
    
        orig_height, orig_width = img.shape[:2]
        inp_images = torch.zeros(len(dets), 3, 224, 224, device=self.device, dtype=torch.float)
        
        for det_idx, det in enumerate(dets):
            bbox = det
            norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                img,
                bbox,
                kp_2d=None ,#if joints2d is None else copy.deepcopy(joints2d[img_idx]),
                scale=1.0,
                crop_size=224
            )
            inp_images[det_idx] = norm_img.float().to(self.device)
    
        try:
            torch.cuda.synchronize()
            start = time.time()
            output = self.model(inp_images)
            torch.cuda.synchronize()
            end = time.time()
            # print('person',end-start)
        except Exception as e:
            import IPython; IPython.embed(); exit()
    
        for k,v in output.items():
            output[k] = v.cpu().numpy()
    
        orig_cam = convert_crop_cam_to_orig_img(
            cam=output['pred_cam'],
            bbox=dets,
            img_width=orig_width,
            img_height=orig_height
        )
    
        smpl_joints2d = convert_crop_coords_to_orig_img(
            bbox=dets,
            keypoints=output['smpl_joints2d'],
            crop_size=224,
        )
    
        output['bboxes'] = dets
        output['orig_cam'] = orig_cam
        output['smpl_joints2d'] = smpl_joints2d
        output['time'] = end-start
            
        poses = []
        for i in range(len(output['pred_pose'][0])):
            res,j = cv2.Rodrigues(output['pred_pose'][0][i])
            poses = poses+res.reshape(3).tolist()
        output['pred_pose'] = poses[:66]
        
        self.num+=1
        self.time+=output['time']
        
        del inp_images
        return output 

class HandNet(Base):
    def __init__(self, ckpt, device, root_path, model_name='hand') -> None:
        super(HandNet, self).__init__(root_path, model_name)
        root=root_path+'/otherwork/Minimal_Hand_pytorch/'
        sys.path.insert(0, '../otherwork/Minimal_Hand_pytorch')
        from Minimal_Hand_pytorch.model.detnet import detnet
        from Minimal_Hand_pytorch.model import shape_net
        from manopth import manolayer
        _mano_root = root+'mano/models'
        module = detnet().to(device)
        check_point = torch.load(ckpt, map_location=device)
        model_state = module.state_dict()
        state = {}
        for k, v in check_point.items():
            if k in model_state:
                state[k] = v
            else:
                print(k, ' is NOT in current model')
        model_state.update(state)
        module.load_state_dict(model_state) 
        
        mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side="right",
                               mano_root=_mano_root,
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')
        
        shape_model = shape_net.ShapeNet(_mano_root=_mano_root)
        shape_net.load_checkpoint(
            shape_model, os.path.join(root+'checkpoints', 'ckp_siknet_synth_41.pth.tar')
        )
        for params in shape_model.parameters():
            params.requires_grad = False
            
        self.module = module
        self.mano = mano.to(device)
        self.device = device
        self.shape_model = shape_model
    # return module, mano

    @torch.no_grad()
    def __call__(self, frame, bboxes):
        sys.path.insert(0, '../otherwork/Minimal_Hand_pytorch')
        from Minimal_Hand_pytorch.utils import func, bone, AIK, smoother

        batch_size = sum([bbox['bbox'][-1]>0 for bbox in bboxes])
        if batch_size ==0:
            return {'poses':[0.0]*90, 'hand_shape':[],'time':0}
        images_tensor = torch.zeros((batch_size, 3, 128, 128), device=self.device)  # (height, width)
        count = 0
        use=[]
        for ann in bboxes:
            bbox = ann['bbox']
            flip = ann['fliplr']
            if bbox[-1]==0:
                continue
            use.append(flip)
            img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
            
            if flip:
                img = cv2.flip(img, 1)
                
            shape_fliter = smoother.OneEuroFilter(4.0, 0.0)
            input = np.flip(img.copy(), -1)
    
            if input.shape[0] > input.shape[1]:
                margin = (input.shape[0] - input.shape[1]) // 2
                input = input[margin:input.shape[0]-margin]
            else:
                margin = (input.shape[1] - input.shape[0]) // 2
                input = input[:, margin:input.shape[1]-margin]
            img = input.copy()
            img = np.flip(img, -1)

            try:
                input = cv2.resize(input, (128, 128))
            except:
                breakpoint()
            input = torch.tensor(input.transpose([2, 0, 1]), dtype=torch.float, device=self.device)  # hwc -> chw
            input = func.normalize(input, [0.5, 0.5, 0.5], [1, 1, 1])
    
            images_tensor[count] = input.to(self.device)
            count += 1
    
        #module
        torch.cuda.synchronize()
        start = time.time()
        # result = module(input.unsqueeze(0))
        result = self.module(images_tensor)
        # torch.cuda.synchronize()
        # end = time.time()
        
        pre_joints = result['xyz']
        pre_joints = pre_joints.clone().detach().cpu().numpy()
        opt_tensor_shape_tensor = torch.zeros((batch_size, 10), device=self.device)
        for ct in range(pre_joints.shape[0]):
            pre_useful_bone_len = bone.caculate_length(pre_joints[ct], label="useful")
        
            shape_model_input = torch.tensor(pre_useful_bone_len, dtype=torch.float)
            shape_model_input = shape_model_input.reshape((1, 15))        
        
            #shape model
            # torch.cuda.synchronize()
            # shape_start = time.time()
            dl_shape = self.shape_model(shape_model_input)
            # torch.cuda.synchronize()
            # shape_end = time.time()
        
            dl_shape = dl_shape['beta'].numpy()
            dl_shape = shape_fliter.process(dl_shape)
            opt_tensor_shape = torch.tensor(dl_shape, dtype=torch.float).to(self.device)
            opt_tensor_shape_tensor[ct] = opt_tensor_shape.reshape(10)
            
        pose0 = torch.eye(3).repeat(batch_size, 16, 1, 1)
        pose0 = pose0.to(self.device)
        #mano
        # torch.cuda.synchronize()
        # mano_start = time.time()
        _, j3d_p0_ops = self.mano(pose0, opt_tensor_shape_tensor)
        # torch.cuda.synchronize()
        # mano_end = time.time()
        res_pose = []
        for ct in range(pre_joints.shape[0]):
            pre_joint = pre_joints[ct]
            template = j3d_p0_ops[ct].cpu().numpy() / 1000.0  # template, m 21*3
    
            ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joint[9] - pre_joint[0])
            j3d_pre_process = pre_joint * ratio  # template, m
            j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
    
        # torch.cuda.synchronize()
        # ik_start = time.time()
            pose_R = AIK.adaptive_IK(template, j3d_pre_process)
            pose_R = torch.from_numpy(pose_R).float()
            res_pose.append(pose_R.cpu().numpy())
            
        torch.cuda.synchronize()
        ik_end = time.time()
                
        data = {'poses':[], 'hand_shape':opt_tensor_shape.float(),
                'time':ik_end - start}
        for ct in range(len(res_pose)):
            pose_R = res_pose[ct]
            poses = []
            for i in range(1, len(pose_R[0])):
                res,j = cv2.Rodrigues(pose_R[0][i])
                poses = poses+res.reshape(3).tolist()
            if use[ct]:
                pose = np.array(poses).reshape((-1,3))
                pose[:, 1::3] = -pose[:, 1::3]
                pose[:, 2::3] = -pose[:, 2::3]
                poses = pose.reshape(45).tolist()
            data['poses'] = data['poses'] + poses
        if len(use)==1:
            if use[0]:
                data['poses'] = data['poses'] + [0.0]*45
            else:
                data['poses'] = [0.0]*45 + data['poses']
                
        self.num+=1
        self.time+=data['time']
        
        return data



def test_onnx():
    import onnx

    onnx_model = onnx.load('/home/xxx/datasets/realtime_test_out/onnx/body_net_pare.onnx') # 加载 onnx 模型
    onnx.checker.check_model(onnx_model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    import onnxruntime
    from onnxruntime.datasets import get_example
    device = torch.device('cuda:0')
    # dummy_input = torch.randn(1, 3,224,224, requires_grad=True).to(device)
    # 测试数据
    path = '/home/xxx/datasets/realtime_test_out/onnx/body_net_pare.onnx'
    example_model = get_example(path)
    sess = onnxruntime.InferenceSession(example_model)
    torch.cuda.synchronize()
    t1 = time.time()
    dummy_input = torch.zeros(1, 3,224,224, requires_grad=True).to(device)
    

    
    
    # onnx 网络输出
    onnx_out = sess.run(None, {'modelInput': to_numpy(dummy_input)})
    torch.cuda.synchronize()
    print("Time:", time.time() - t1)


def main(yolo_ckpt, body_ckpt, hand_ckpt, args, root_path):
    
    device = torch.device('cuda')
    # model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    
    body_net = BodyNet(body_ckpt, device, root_path)
    # hand_net = HandNet(hand_ckpt, device, root_path)
    
    
    
    
    # dummy_input = torch.randn(1, input_size, requires_grad=True)  
    dummy_input = torch.randn(1, 3,224,224, requires_grad=True).to(device)
    
    # inp_images = torch.zeros(len(dets), 3, 224, 224, device=self.device, dtype=torch.float)
    # Export the model   
    print('cvt to onnx')
    torch.onnx.export(body_net.model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "/home/xxx/datasets/realtime_test_out/onnx/pare_new.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

    

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--usb', action='store_true')
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.getcwd(),'..'))

    body_ckpt = root + '/otherwork/PARE/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
    hand_ckpt = root + '/otherwork/Minimal_Hand_pytorch/new_check_point/bmc_ckp.pth'
    yolo_ckpt = root + '/otherwork/yolo-best.pt'
    # client = BaseSocketClient(args.host, args.port)

    main(yolo_ckpt=yolo_ckpt, body_ckpt=body_ckpt, hand_ckpt=hand_ckpt, args=args, root_path=root)

    #python .\apps\vis\vis_server.py --cfg .\config\vis3d\o3d_scene_smplh.yml --opts hostport 0.0.0.0:9999
    #$env:HTTP_PROXY="http://10.0.0.10:7890"
    #python .\realtime_infer_bodyandhand.py --usb