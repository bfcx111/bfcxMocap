'''
  @ Date: 2022-04-13 17:49:25
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-06-01 22:50:24
  @ FilePath: /EasyMocapPublic/apps/camera/realtime_infer.py
'''

import torch
import cv2
import numpy as np
from torchvision.transforms import transforms

from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.mytools.utils import Timer
from easymocap.estimator.topdown_wrapper import BaseTopDownHeatmapWrapper
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def run_cmd(cmd, verbo=True, bg=False):
    print('[run] ' + cmd, 'run')
    os.system(cmd)
    return []


def rescalebox(box,rescale=1.4):
    c = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    w = (box[2]-box[0]) * rescale
    h = (box[3]-box[1]) * rescale
    bbox=[
        c[0]-w/2,
        c[1]-h/2,
        c[0]+w/2,
        c[1]+h/2,
        box[4]
        ]
    return bbox
def main(yolo_ckpt, args):
    device = torch.device('cuda',args.gpu)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    # model.eval()
    min_detect_thres = 0

    if False:
        cap = cv2.VideoCapture(2)
        w, h = 1280, 720
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 60)
        def get_image():
            return cap.read()
    else:
        count = 0
        from os.path import join
        import os

        # root = '/nas/dataset/HalfCapture/process0614/220614-0_4+001100+010600/images/02'
        root = '/dellnas/dataset/HalfCapture/process0617/yolo0622/person0_step60/images/03'
        def get_image():
            nonlocal count
            imgname = join(root, '{:06d}.jpg'.format(count))
            if not os.path.exists(imgname):
                return False, None
            count += 1
            return True, cv2.imread(imgname)
    Id=0

    while True:
        with Timer('imread'):
            flag, frame = get_image()
            # frame = frame / 255
            if flag ==False:
                break
            # frame = cv2.imread(join(root, i))
        with Timer('detect'):
            results = model(frame)

        # frame = frame * 255
        # breakpoint()
        arrays = np.array(results.pandas().xyxy[0])
        results = {}
        for i, res in enumerate(arrays):
            name = res[6] + str(i)
            bbox = res[:5]
            # bbox = rescalebox(bbox)
            if bbox[4] < min_detect_thres:
                continue
            if name not in results.keys():
                results[name] = bbox
            else:
                if results[name][-1] < bbox[-1]:
                    results[name] = bbox
        vis = frame.copy()

        # results = res
        # if 'handl' in results.keys() or 'handr' in results.keys(): 
        #     if 'handl' not in results.keys():
        #         results['handl'] = [0, 0, 100, 100, 0]
        #     if 'handr' not in results.keys():
        #         results['handr'] = [0, 0, 100, 100, 0]

        for name, bbox in results.items():
            plot_bbox(vis, bbox, name)
            

        if flag:
            # cv2.imwrite('/nas/users/wangjunhao/out/test_yolo/{}.jpg'.format(Id),vis)
            cv2.imshow('vis', vis)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            # if Id==10:
            #     break
            # break
            Id+=1
        else:
            break

    Timer.report()

def clean_ckpt(ckptname, newname):
    data = torch.load(ckptname)['state_dict']
    output = {}
    for key, val in data.items():
        if key.startswith('model.'):
            output[key[6:]] = val
    torch.save(output, newname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()
    # yolo_ckpt = '/home/xxx/share/humanmodels/yolo-20220608-best.pt'
    # hand_ckpt = '/home/xxx/share/humanmodels/last_interhand_coco.ckpt'
    # yolo_ckpt = '/nas/users/shuaiqing/EasyMocapPublic/3rdparty/yolov5/runs/train/yolo-0608/weights/best.pt'
    # hand_ckpt = '/home/shuaiqing/EasyMocapPublic/easypose/hrnet+hand/model/last.ckpt'

    yolo_ckpt = '/nas/users/wangjunhao/out/yolo-best1.pt'
    main(yolo_ckpt=yolo_ckpt, args=args)