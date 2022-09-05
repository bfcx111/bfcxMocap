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


def test_onnx():
    gpu_id=0
    import onnx

    onnx_model = onnx.load('/home/xxx/datasets/realtime_test_out/onnx/body_net_pare.onnx') # 加载 onnx 模型
    onnx.checker.check_model(onnx_model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    import onnxruntime
    from onnxruntime.datasets import get_example
    device = torch.device('cuda:{}'.format(gpu_id))
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
    gpu_id=0
    device = torch.device('cuda:{}'.format(gpu_id))
    model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt).to(device)
    
    # body_net = BodyNet(body_ckpt, device, root_path)
    # hand_net = HandNet(hand_ckpt, device, root_path)
    
    
    
    # breakpoint()
    # dummy_input = torch.randn(1, input_size, requires_grad=True)  
    dummy_input = torch.zeros(1, 3,640,640, requires_grad=True).to(device)
    
    # inp_images = torch.zeros(len(dets), 3, 224, 224, device=self.device, dtype=torch.float)
    # Export the model   
    print('cvt to onnx')
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "/home/xxx/datasets/realtime_test_out/onnx/yolo_0729.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=16,    # the ONNX version to export the model to 
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
    yolo_ckpt = '/home/xxx/share/yolov5-20220701-best.pt'
    # client = BaseSocketClient(args.host, args.port)

    main(yolo_ckpt=yolo_ckpt, body_ckpt=body_ckpt, hand_ckpt=hand_ckpt, args=args, root_path=root)

    #python .\apps\vis\vis_server.py --cfg .\config\vis3d\o3d_scene_smplh.yml --opts hostport 0.0.0.0:9999
    #$env:HTTP_PROXY="http://10.0.0.10:7890"
    #python .\realtime_infer_bodyandhand.py --usb