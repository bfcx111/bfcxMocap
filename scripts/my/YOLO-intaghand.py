'''
  @ Date: 2022-05-09 17:49:25
  @ Author: wjh
'''
# 这个脚本用于快速的单个USB相机测试网络
# 单目实时半身运动捕捉 
# 1. yolo
# 2. Hand4Whole
# 3. smplh

import torch
import cv2
import numpy as np
from tqdm import tqdm
from easymocap.mytools.utils import Timer

from torchvision.transforms import transforms
from easymocap.socket.base_client import BaseSocketClient

import sys
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, '../otherwork/Hand4Whole_RELEASE')
# sys.path.insert(0, osp.join('../otherwork/Hand4Whole_RELEASE', 'main'))
# sys.path.insert(0, osp.join('../otherwork/Hand4Whole_RELEASE', 'data'))
# sys.path.insert(0, osp.join('../otherwork/Hand4Whole_RELEASE', 'common'))
from main.model import get_model

from common.utils.preprocessing import process_bbox, generate_patch_image
from common.utils.human_models import smpl_x
from common.utils.vis import render_mesh  #, save_obj

class Model:
    def __init__(self, model, ckpt, device, input_size, output_size, use_normalize=False) -> None:
        # model.to(device)
        self.model = model
        model.load_state_dict(torch.load(ckpt, map_location=device)['network'], strict=False)
        model.eval()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        if use_normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.ToTensor()
        self.show = False
    
    def preprocess(self, ):
        pass

    def __call__(self, images, bboxes):
        original_img_height, original_img_width = images.shape[:2]
        data=[]
        # for image, bbox in zip(images, bboxes):
        id=0
        bbox=bboxes[:4]
        image=images
        # prepare bbox
        bbox[2]-=bbox[0]
        bbox[3]-=bbox[1]
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(image, bbox, 1.0, 0.0, False, self.input_size) 
        img = self.transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = self.model(inputs, targets, meta_info, 'test')


        out['smplx_body_pose']=out['smplx_body_pose'].detach().cpu().numpy()
        out['smplx_lhand_pose']=out['smplx_lhand_pose'].detach().cpu().numpy()
        out['smplx_rhand_pose']=out['smplx_rhand_pose'].detach().cpu().numpy()
        pose_mean=smpl_x.layer['neutral'].pose_mean.detach().cpu().numpy()

        out['smplx_body_pose'][0]+=pose_mean[3:3+len(out['smplx_body_pose'][0])]
        out['smplx_lhand_pose'][0]+=pose_mean[len(pose_mean)-len(out['smplx_rhand_pose'][0])-len(out['smplx_lhand_pose'][0]):len(pose_mean)-len(out['smplx_rhand_pose'][0])]
        out['smplx_rhand_pose'][0]+=pose_mean[len(pose_mean)-len(out['smplx_rhand_pose'][0]):]
        res={}
        res["id"]=id
        res["Rh"]=np.array([0.0,0.0,0.0], dtype=np.float32).reshape(1,3)
        res["Th"]=np.array([0.0,0.0,0.0], dtype=np.float32).reshape(1,3)
        bodypose=[0.0,0.0,0.0]+out['smplx_body_pose'].tolist()[0] + out['smplx_lhand_pose'].tolist()[0] + out['smplx_rhand_pose'].tolist()[0]

        #smpl
        # while len(bodypose)<72:
        #     bodypose.append(0.0)
        #smplx
        # while len(bodypose)<83:
        #     bodypose.append(0.0)
        
        sp=out['smplx_shape'].detach().cpu().numpy().tolist()[0] + [0.0,0.0,0.0,0.0,0.0,0.0]
        res["poses"]=np.array(bodypose, dtype=np.float32).reshape(1,len(bodypose))
        res["expression"]=np.array(out['smplx_expr'].detach().cpu().numpy(), dtype=np.float32)
        res["shapes"]=np.array(sp, dtype=np.float32).reshape(1,len(sp))
        res['meshs'] = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        data.append(res)
        return data

def load_wholebodynet(ckpt, device, name):
    assert osp.exists(ckpt), 'Cannot find model at ' + ckpt
    print('Load checkpoint from {}'.format(ckpt))
    cudnn.benchmark = True
    model = get_model('test')
    model = DataParallel(model).cuda()
    return Model(model, ckpt, device, input_size=(512, 384), output_size=(8, 8, 6))

def main(yolo_ckpt, wholebody_ckpt, name):
    device = torch.device('cuda')
    model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    wholebody_net= load_wholebodynet(wholebody_ckpt, device, name)
    min_detect_thres = 0.3
     
    if True:
        cap = cv2.VideoCapture(0)
        w, h = 720 , 720 # 1280, 720
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 60)

    lasmesh=[]
    #image:
    # PATH='/home/xxx/datasets/desktop/0'
    # for name in tqdm((os.listdir(PATH))):
    #cam
    while True:
        with Timer('imread'):
            flag, frame = cap.read()
            
        #'/nas/users/wangjunhao/otherwork/Hand4Whole_RELEASE/demo/input.png'
        #image:
        # frame=cv2.imread(osp.join(PATH,name))
        # frame=cv2.imread(osp.join(PATH,'000000.jpg')) #083

        with Timer('detect'):
            results = model(frame)
        arrays = np.array(results.pandas().xyxy[0])
        results = {}
        for res in arrays:
            name = res[6]
            bbox = res[:5]
            if bbox[4] < min_detect_thres:
                continue
            if name not in results.keys():
                results[name] = bbox
            else:
                if results[name][-1] < bbox[-1]:
                    results[name] = bbox
        vis = frame.copy()
        
        if 'person' in results.keys():
            with Timer('est body'):
                data = wholebody_net(frame,results['person'])
            
            lasmesh.append(data[0]['poses'])
            if len(lasmesh)>10:
                lasmesh.pop(0)
            data[0]['poses']=sum(lasmesh)/len(lasmesh)

            client.send_smpl(data)
            
        if flag:
            cv2.imshow('vis', vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        else:
            break
    cap.release()
    Timer.report()



if __name__ == '__main__':
    yolo_ckpt = '/home/xxx/share/humanmodels/yolov5-20220411.pt'
    device = torch.device('cuda')
    model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    frame = cv2.imread('')
    results = model(frame)
    arrays = np.array(results.pandas().xyxy[0])
    results = {}
    for res in arrays:
        name = res[6]
        bbox = res[:5]
        if bbox[4] < min_detect_thres:
            continue
        if name not in results.keys():
            results[name] = bbox
        else:
            if results[name][-1] < bbox[-1]:
                results[name] = bbox
    img = frame[int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1,:].copy()
    
    Hand4Whole_ckpt = '/home/xxx/work/otherwork/Hand4Whole_RELEASE/demo/snapshot_6_1.pth.tar'
    main(yolo_ckpt=yolo_ckpt, wholebody_ckpt=Hand4Whole_ckpt, name=model_name_list[-1])