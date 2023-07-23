#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:51:11 2022

@author: xxx
"""

# import torch
import cv2
import numpy as np
import os
import tensorrt as trt
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import argparse

import pycuda.autoinit
import pycuda.driver as cuda
# import tensorrt as trt

TRT_LOGGER = trt.Logger()
# engine_file_path = "/home/xxx/datasets/realtime_test_out/onnx/pare.trt"

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


import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)



network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
# model_path = '/home/xxx/datasets/realtime_test_out/onnx/body_net_pare.onnx'
# success = parser.parse_from_file(model_path)


# for idx in range(parser.num_errors):
#     print(parser.get_error(idx))

# if not success:
#     pass # Error handling code here
    
# config = builder.create_builder_config()

# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
# serialized_engine = builder.build_serialized_network(network, config)
# with open("sample.engine", "wb") as f:
#     f.write(serialized_engine)

# engine_path = '/home/xxx/datasets/realtime_test_out/onnx/pare.trt'
#engine_path = '/home/xxx/datasets/realtime_test_out/onnx/hrnet.trt'
# engine_path = '/home/xxx/datasets/realtime_test_out/onnx/hrnet8221.trt'
#engine_path = '/home/xxx/datasets/realtime_test_out/onnx/hrnet2.engine'
engine_path = '/dellnas/users/wangjunhao/data/hrnet16.engine'
engine_path = '/dellnas/users/wangjunhao/data/hand-hrnet16.engine'
with open(engine_path, "rb") as f:
    serialized_engine = f.read()
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)



context = engine.create_execution_context()



inputs, outputs, bindings, stream = allocate_buffers(engine)
print(inputs, outputs, bindings, stream)
print(engine_path)
# 前处理部分
# dummy_input = torch.zeros((3, 3, 384, 288)).detach().numpy()
# dummy_input = torch.zeros((1, 3, 384, 288)).detach().numpy()
dummy_input = np.ones((3, 3, 384, 288))
dummy_input = np.ones((6, 3, 256, 256))
for i in range(20):
    
    dummy_input = dummy_input.astype(np.float32)
    inputs[0].host = dummy_input
    # torch.cuda.synchronize()
    t1 = time.time()
    # dummy_input = torch.zeros(1, 3,224,224, requires_grad=True).detach().numpy()
    # dummy_input = torch.zeros((3, 3, 384, 288), device=device)

    # 前处理结束

    # 开始推理
    # inputs[0].host = cv_images
    
    trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # breakpoint()
    # print(trt_outputs.shape)
    # torch.cuda.synchronize()
    print("Time:", time.time() - t1)



# input_idx = engine[input_name]
# output_idx = engine[output_name]

# buffers = [None] * 2 # Assuming 1 input and 1 output
# buffers[input_idx] = input_ptr
# buffers[output_idx] = output_ptr

# context.execute_async_v2(buffers, stream_ptr)





















# with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, \
#         runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
            
            

#     inputs, outputs, bindings, stream = allocate_buffers(engine)
#     print(inputs, outputs, bindings, stream)

#     # 前处理部分
#     torch.cuda.synchronize()
#     t1 = time.time()
#     dummy_input = torch.zeros(1, 3,224,224, requires_grad=True).detach().numpy()
    
#     # cv_img = cv2.imread("image030.jpg")
#     # cv_img = cv2.resize(cv_img, (512, 512), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
#     # cv_images = cv_img.astype('float32')
#     # cv_images = cv_images / 127.5 - 1  # 归一化到 -1 到 1 之间
#     # cv_images = np.expand_dims(cv_images, 0)
#     # 前处理结束

#     # 开始推理
#     # inputs[0].host = cv_images
#     inputs[0].host = dummy_input
#     trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#     # print(trt_outputs.shape)
#     torch.cuda.synchronize()
#     print("Time:", time.time() - t1)

# 	# 由于trt_outputs是一个一维的list，而我们需要的输出是(1, 512, 512, 2)的张量，所以要整形
#     trt_output = [output.reshape(shape) for output, shape in zip(trt_outputs, [(1, 512, 512, 2)])]


# poses = torch.tensor(trt_outputs[3]).reshape((1,24,3,3))
