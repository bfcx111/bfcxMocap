'''
  @ Author: wjh
'''

import torch
import cv2
import numpy as np

from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.mytools.utils import Timer
from easymocap.estimator.topdown_wrapper import BaseTopDownHeatmapWrapper

def load_bodynet(ckpt, device):
    from easymocap.estimator.HRNet.hrnet import HRNet
    config_hrnet = {
        'nof_joints': 17,
        'c': 48,
        'ckpt': 'data/models/pose_hrnet_w48_384x288.pth',
        'input_size': [384, 288],
        'output_size': [96, 72]
    }
    model = HRNet(c=config_hrnet['c'], nof_joints=config_hrnet['nof_joints']).to(device)
    return BaseTopDownHeatmapWrapper(model, ckpt, device, input_size=config_hrnet['input_size'], output_size=config_hrnet['output_size'], name='hrnet')

def load_handnet(ckpt, device):
    from sensehand_demo.models.resnet_deconv.resnet_deconv import ResNet_Deconv
    model = ResNet_Deconv()
    return BaseTopDownHeatmapWrapper(model, ckpt, device, input_size=(256, 256), output_size=(64, 64), use_normalize=False, name='handnet')

def main(yolo_ckpt, hand_ckpt, body_ckpt):
    device = torch.device('cuda')
    # model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    body_net = load_bodynet(body_ckpt, device)
    # hand_net = load_handnet(hand_ckpt, device)
    dummy_input = torch.zeros((3, 3, 384, 288), device=device)
    # dummy_input = np.zeros((384, 288,3)).astype(np.float32)
    # # body_net(dummy_input,[{'bbox':[0,0,288,288,1]},{'bbox':[0,0,288,288,1]},{'bbox':[0,0,288,288,1]}])
    # body_net([dummy_input], [[{'bbox': [0,0,70,100,1], 'rot': 0, 'fliplr': False}]])
    import time

    while True:
        torch.cuda.synchronize()
        t1 = time.time()
        out = body_net.model(dummy_input)
        torch.cuda.synchronize()
        print("Time:", time.time() - t1)
    # torch.onnx.export(body_net.model,         # model being run 
    #     dummy_input,       # model input (or a tuple for multiple inputs) 
    #     f"/home/xxx/datasets/realtime_test_out/onnx/hrnet2.onnx",       # where to save the model  
    #     export_params=True,  # store the trained parameter weights inside the model file 
    #     opset_version=10,    # the ONNX version to export the model to 
    #     do_constant_folding=True,  # whether to execute constant folding for optimization 
    #     input_names = ['modelInput'],   # the model's input names 
    #     output_names = ['modelOutput'], # the model's output names 
    #     dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
    #         'modelOutput' : {0 : 'batch_size'}})    
    
    


if __name__ == '__main__':
    # yolo_ckpt = '/nas/share/humanmodels/yolov5-wholebody-220428.pt'
    # yolo_ckpt = '/nas/share/humanmodels/yolov5-wholebody-220428.pt'
    yolo_ckpt = '/nas/users/shuaiqing/EasyMocapPublic/3rdparty/yolov5/runs/train/exp/weights/best.pt'
    hand_ckpt = '/nas/share/humanmodels/hand_resnet_deconv.ckpt'
    body_ckpt = 'data/models/pose_hrnet_w48_384x288.pth'
    main(yolo_ckpt=yolo_ckpt, hand_ckpt=hand_ckpt, body_ckpt=body_ckpt)