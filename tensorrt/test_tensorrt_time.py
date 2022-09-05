
import torch
# import torch.backends.cudnn as cudnn

# import cv2
import numpy as np
import time
# import json
# from multiprocessing import Process, Queue
# from copy import deepcopy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp

# from easymocap.socket.base_client import BaseSocketClient
# from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
# from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, get_affine_transform
# from easymocap.multistage.torchgeometry import axis_angle_to_euler, euler_to_axis_angle
# from easymocap.annotator.file_utils import save_json

# from my_realtime_need_file import get_single_image_crop_demo

# from os.path import join
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda
# import tensorrt as trt

# import sys
# from scipy.spatial.transform import Rotation as R
# from manopth.manolayer import ManoLayer
# from scipy.spatial import transform as tt #Rotation as R
# import scipy
# import scipy


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def test_engine_time(engine_path, input_shape):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    print('engine: ', engine_path)
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # print(inputs, outputs, bindings, stream)

    dummy_input = np.random.rand(*input_shape).astype(np.float16)
    print('>>> Input shape: ', dummy_input.shape)
    total_time = 0
    N_repeat = 100
    for i in range(N_repeat):
        inputs[0].host = dummy_input
        torch.cuda.synchronize()
        t1 = time.time()

        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        torch.cuda.synchronize()
        total_time += time.time() - t1

    print(">>> Time: {:.3f}ms".format(total_time/N_repeat*1000))


if __name__ == '__main__':
    config = {
        'mesh2pose': {
            'path': '/home/xxx/datasets/realtime_test_out/onnx/mesh2pose.engine',
            'shape': (2,778,3)
        },
        'yolo': {
            'path': 'D:/Work/real-time-bodyandhand/onnx/yolo_fp16.engine',
            'path': '/home/xxx/share/yolo-0810-best.engine',
            'shape': (1, 3, 640, 640)
        },
        'pare': {
            # 'path': 'D:/Work/real-time-bodyandhand/onnx/pare_fp16.trt',
            'path': '/home/xxx/datasets/realtime_test_out/onnx/pare_0812_fp16.engine',
            'shape': (1, 3, 224, 224)
        },
        'hand': {
            # 'path': 'D:/Work/real-time-bodyandhand/onnx/handmesh_2hand_fp16.engine',
            'path': '/home/xxx/datasets/realtime_test_out/onnx/handmesh_2hand_fp16.engine',
            'shape': (2, 3, 128, 128)
        },       
    }
    for key, val in config.items():
        test_model(val['path'], val['shape'])
        