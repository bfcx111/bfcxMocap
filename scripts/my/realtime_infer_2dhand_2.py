'''
  @ Date: 2022-04-13 17:49:25
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-06-01 22:50:24
  @ FilePath: /EasyMocapPublic/apps/camera/realtime_infer.py
'''
# 这个脚本用于快速的单个USB相机测试网络
# 例如：
# 1. yolo
# 2. pose2d
# 3. pose3d
import torch
import cv2
import numpy as np
from torchvision.transforms import transforms

from easymocap.mytools.vis_base import plot_bbox, plot_keypoints_auto
from easymocap.mytools.utils import Timer
from easymocap.estimator.topdown_wrapper import BaseTopDownHeatmapWrapper
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
    # from sensehand_demo.models.resnet_deconv.resnet_deconv import ResNet_Deconv
    # easymocap.easypose.model.pose_hrnet.PoseHighResolutionNet
    from easymocap.config import load_object, Config
    cfg = Config.load('/nas/users/wangjunhao/EasyMocapPublic/config/easypose/network/hrnet_twohand.yml')
    model = load_object(cfg.network_module, cfg.network_args)
    # model = ResNet_Deconv()
    return BaseTopDownHeatmapWrapper(model, ckpt, device, input_size=(256, 256), output_size=(64, 64), use_normalize=True, name='handnet')

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
def main(yolo_ckpt, hand_ckpt):
    device = torch.device('cuda')
    model = torch.hub.load('ultralytics/yolov5', 'custom', yolo_ckpt)
    hand_net = load_handnet(hand_ckpt, device)
    # body_net = load_bodynet(body_ckpt, device)
    min_detect_thres = 0.3
    # print("IMG_ID:  ",IMG_ID)
    # IMG_ID='2'
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
        # root = '/home/xxx/datasets/desktop/0608/images/'+IMG_ID
        # root = '/home/xxx/datasets/clip/3saU5racsGE+001750+003390'
        # root = '/home/xxx/datasets/clip/0UakYLTiplc+007820+009110'
        # root = '/home/xxx/datasets/clip/0UakYLTiplc+004040+004650'
        root = '/nas/dataset/HalfCapture/process0614/220614-0_4+001100+010600/images/02'
        def get_image():
            nonlocal count
            imgname = join(root, '{:06d}.jpg'.format(count))
            if not os.path.exists(imgname):
                return False, None
            count += 1
            return True, cv2.imread(imgname)
    Id=0
    # root = '/nas/dataset/human/InterHand2.6M/InterHand2.6M_30fps_batch1/images/test/Capture0/ROM01_No_Interaction_2_Hand/cam400262'
    # L = os.listdir(root)
    # L.sort()
    # for i in L:
    while True:
        with Timer('imread'):
            flag, frame = get_image()
            if flag ==False:
                break
            # frame = cv2.imread(join(root, i))
        with Timer('detect'):
            results = model(frame)
        arrays = np.array(results.pandas().xyxy[0])
        results = {}
        for i, res in enumerate(arrays):
            name = res[6] + str(i)
            bbox = res[:5]
            bbox = rescalebox(bbox)
            if bbox[4] < min_detect_thres:
                continue
            if name not in results.keys():
                results[name] = bbox
            else:
                if results[name][-1] < bbox[-1]:
                    results[name] = bbox
        vis = frame.copy()
        # if 'person' in results.keys():
        #     with Timer('est body'):
        #         pts = body_net([frame], [[{'bbox': results['person'], 'rot': 0, 'fliplr': False}]])
        #     pts = coco17tobody25(pts)
        #     plot_keypoints_auto(vis, pts[0], pid=0)
        # import pdb;pdb.set_trace()
        # res={}
        # for k in results.keys():
        #     if 'person' in k:
        #         res[k] = results[k]
        #     if 'handl' in k and 'handl' not in res:
        #         res['handl'] = results[k]
        #     if 'handr' in k and 'handr' not in res:
        #         res['handr'] = [400., 200., 1550., 900., 1]
                # res['handr'] = results[k]
        results = res
        if 'handl' in results.keys() or 'handr' in results.keys(): 
            if 'handl' not in results.keys():
                results['handl'] = [0, 0, 100, 100, 0]
            if 'handr' not in results.keys():
                results['handr'] = [0, 0, 100, 100, 0]
            with Timer('est hand'):
                pts = hand_net([frame], [[{'bbox': results['handl'], 'rot': 0, 'fliplr': False}, {'bbox': results['handr'], 'rot': 0, 'fliplr': False}]])
            # import pdb;pdb.set_trace()
            # pts[0][21:,:] = pts[1][21:,:]
            plot_keypoints_auto(vis, pts[0], pid=0)
            plot_keypoints_auto(vis, pts[1], pid=0)
            # pts[1][:21,:] = 0
            # if 'handr' in results.keys() and 'handl' in results.keys():
            #     pts[0][21:,:] = pts[1][21:,:]
            #     plot_keypoints_auto(vis, pts[0], pid=0)
            # else:
            #     if 'handly' in results.keys():
            #         plot_keypoints_auto(vis, pts[0], pid=0)
            #     if 'handr' in results.keys():
            #         plot_keypoints_auto(vis, pts[1], pid=0)
        for name, bbox in results.items():
            plot_bbox(vis, bbox, name)
            
        # for name, bbox in res.items():
            # plot_bbox(vis, bbox, name)
            # print(name,bbox)
        if flag:
            cv2.imshow('vis', vis)
            # cv2.imwrite('/home/xxx/datasets/out/test'+IMG_ID+'/%06d.jpg'%Id,vis)
            # cv2.imwrite('/home/xxx/datasets/out/renzhe/%06d.jpg'%Id,vis)
            Id+=1
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            # if Id==10:
            #     break
            # break
        else:
            break
    # cap.release()

    # cmd = 'ffmpeg -r 25 -i /home/xxx/datasets/out/yolo_hand'+IMG_ID+'/%06d.jpg -vcodec libx264 -r 25 /home/xxx/datasets/out/hand_test'+IMG_ID+'.mp4'
    # cmd = 'ffmpeg -r 25 -i /home/xxx/datasets/out/renzhe/%06d.jpg -vcodec libx264 -r 25 /home/xxx/datasets/out/jieyin.mp4'

    # run_cmd(cmd)
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
    # parser.add_argument('IMG_ID', type=str)
    args = parser.parse_args()
    # yolo_ckpt = '/home/xxx/share/humanmodels/yolo-20220608-best.pt'
    # hand_ckpt = '/home/xxx/share/humanmodels/last_interhand_coco.ckpt'
    yolo_ckpt = '/nas/users/shuaiqing/EasyMocapPublic/3rdparty/yolov5/runs/train/yolo-0608/weights/best.pt'
    hand_ckpt = '/home/shuaiqing/EasyMocapPublic/easypose/hrnet+hand/model/last.ckpt'
    main(yolo_ckpt=yolo_ckpt, hand_ckpt=hand_ckpt)