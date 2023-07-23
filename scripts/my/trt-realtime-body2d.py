import cv2
import numpy as np
import time

import os

import torch
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit

from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, get_affine_transform


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
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

class BaseTopDownHeatmapWrapper:
    def __init__(self, model, input_size, output_size, use_normalize=True, name='topdown') -> None:
        self.input_size = input_size
        self.output_size = output_size
        if use_normalize:
            self.normalize_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
            self.normalize_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
            # self.transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
        else:
            self.normalize_mean = np.zeros((1, 1, 1, 3))
            self.normalize_std = np.ones((1, 1, 1, 3))
        self.normalize_mean = self.normalize_mean.transpose(0,3, 2, 1)
        self.normalize_std = self.normalize_std.transpose(0,3, 2, 1)
        self.show = False
        self.model = model

    def emplace_data(self, images, bboxes):
        # crop image
        batch_size = sum([len(bbox) for bbox in bboxes])
        # numpy
        # images_tensor = torch.zeros((batch_size, 3, self.input_size[1], self.input_size[0]), device=self.device)  # (height, width)
        images_numpy = np.zeros((batch_size, 3, self.input_size[1], self.input_size[0]))#, device=self.device)  # (height, width)
        count = 0
        infos = {'center': [], 'scale': [], 'rot': [], 'flip': []}
        for i_img, (image, bbox) in enumerate(zip(images, bboxes)):
            for ibox, box in enumerate(bbox):
                # breakpoint()
                _box = box['bbox']
                rot = box['rot']
                center, scale = box_to_center_scale(_box, self.input_size[0], self.input_size[1])
                infos['center'].append(center)
                infos['scale'].append(scale)
                infos['rot'].append(rot)
                trans = get_affine_transform(center, scale, rot=rot, output_size=self.input_size)
                # model_input = cv2.warpAffine(
                #     image, trans,
                #     (int(self.input_size[0]), int(self.input_size[1])),
                #     flags=cv2.INTER_LINEAR)
                # if box['fliplr']:
                #     model_input = cv2.flip(model_input, 1)
                # infos['flip'].append(box['fliplr'])
                model_input= image
                if self.show:
                    cv2.imshow('input_{}_{}'.format(i_img, ibox), model_input)
                    cv2.waitKey(10)
                # model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
                # channel3->1
                # breakpoint()
                model_input = model_input.transpose(0,3, 2, 1)
                images_numpy[count] = model_input#.to(self.device)
                count += 1
        images_numpy = (images_numpy - self.normalize_mean)/self.normalize_std
        self.model.emplace_data(images_numpy)
        
    def get_output(self):
        out = self.model.get_output()
        # if self.show:
        #     for i in range(out.shape[0]):
        #         feat = out[i].sum(axis=0)
        #         feat = (feat/feat.max()*255).astype(np.uint8)
        #         cv2.imshow('heatmap{}'.format(i), feat)
        #         cv2.waitKey(30)
        # coords, max_val = get_final_preds(out, infos['center'], infos['scale'], infos['rot'], infos['flip'])
        # pts = np.concatenate((coords, max_val), axis=2)
        # pts[max_val[..., 0]<0.1] = 0.
        # return pts

class TrtModel():
    def __init__(self,engine_path) -> None:
        # self.img_id = 0
        # self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        # engine_path = '/home/xxx/datasets/realtime_test_out/onnx/pare_new.trt'
        # engine_path = '/home/xxx/datasets/realtime_test_out/onnx/pare_test0728-1.trt'
        # engine_path = '/home/xxx/datasets/realtime_test_out/onnx/hrnet16.engine'
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
        pass

    def emplace_data(self, data):
        self.inputs[0].host = data

    def get_output(self):
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # do_inferencexxx
        return trt_outputs

if __name__ == '__main__':
    engine_path = '/dellnas/users/wangjunhao/data/hrnet16.engine'

    trtmodel = TrtModel(engine_path)
    wrapper1 = BaseTopDownHeatmapWrapper(trtmodel, [384,288], [96,72])
    wrapper2 = BaseTopDownHeatmapWrapper(trtmodel, [384,288], [96,72])
    data = np.random.rand(0,255,(1,384,288,3)).astype(np.float32)
    # pipeline 0
    bboxs = [[{'bbox':[0,0,288,288,1],'rot':0}],[{'bbox':[0,0,288,288,1],'rot':0}],[{'bbox':[0,0,288,288,1],'rot':0}]]
    torch.cuda.synchronize()
    t1 = time.time()
    wrapper1.emplace_data([data,data,data],bboxs)
    wrapper1.get_output()
    wrapper2.emplace_data([data,data,data],bboxs)    
    wrapper2.get_output()
    torch.cuda.synchronize()
    t2 = time.time()
    print('time1:',t2-t1)

    # pipeline 1
    torch.cuda.synchronize()
    t3 = time.time()
    wrapper1.emplace_data([data,data,data],bboxs)
    wrapper2.emplace_data([data,data,data],bboxs)    
    wrapper1.get_output()
    wrapper2.get_output()    
    torch.cuda.synchronize()
    t4 = time.time()
    print('time1:',t4-t3)


