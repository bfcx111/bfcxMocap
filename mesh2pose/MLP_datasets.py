import torch.utils.data as data
from easymocap.annotator.file_utils import getFileList, save_json, save_annot
from pathlib import Path
import os
import cv2
import json
import numpy as np
import random

import torch
from easymocap.estimator.HRNet.hrnet_api import box_to_center_scale, get_affine_transform

# import skimage.io as io

def json_load(p):
    # _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d
def read_mask_woclip(idx, base_path, set_name):
    mask_path = os.path.join(base_path, set_name, 'mask',
                             '%08d.jpg' % idx)
    # _assert_exist(mask_path)
    return cv2.imread(mask_path)[:, :, 0]


def load_db_annotation(base_path, writer=None, set_name=None):
    if set_name in ['training', 'train']:
        # if writer is not None:
        #     writer.print_str('Loading FreiHAND training set index ...')
        # t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % 'training')

        # assumed paths to data containers
        mano_path = os.path.join(base_path, '%s_mano.json' % 'training')
        xyz_path = os.path.join(base_path, '%s_xyz.json' % 'training')

        # load if exist
        K_list = json_load(k_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        # if writer is not None:
        #     writer.print_str('Loading of %d %s samples done in %.2f seconds' % (len(K_list), set_name, time.time()-t))
        return zip(K_list, mano_list, xyz_list)
    elif set_name in ['evaluation', 'eval', 'val', 'test']:
        # if writer is not None:
        #     writer.print_str('Loading FreiHAND eval set index ...')
        # t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % 'evaluation')
        scale_path = os.path.join(base_path, '%s_scale.json' % 'evaluation')
        K_list = json_load(k_path)
        scale_list = json_load(scale_path)

        assert len(K_list) == len(scale_list), 'Size mismatch.'
        # if writer is not None:
        #     writer.print_str('Loading of %d eval samples done in %.2f seconds' % (len(K_list), time.time() - t))
        return zip(K_list, scale_list)
    else:
        raise Exception('set_name error: ' + set_name)



def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)
def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift, shift_wh=None, inv=False, return_shift=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    if shift_wh is not None:
        shift_lim = (max((src_w - shift_wh[0]) / 2, 0), max((src_h - shift_wh[1]) / 2, 0))
        x_shift = shift[0] * shift_lim[0]
        y_shift = shift[1] * shift_lim[1]
    else:
        x_shift = y_shift = 0
    src_center = np.array([c_x + x_shift, c_y + y_shift], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    if return_shift:
        return trans, [x_shift/src_w, y_shift/src_h]
    return trans

def get_m1to1_gaussian_rand(scale):
    r = 2
    while r < -1 or r > 1:
        r = np.random.normal(scale=scale)

    return r
def get_aug_config(exclude_flip, base_scale=1.1, scale_factor=0.25, rot_factor=60, color_factor=0.2, gaussian_std=1):
    # scale_factor = 0.25
    # rot_factor = 60
    # color_factor = 0.2

    # scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    scale = get_m1to1_gaussian_rand(gaussian_std) * scale_factor + base_scale
    # rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
    rot = get_m1to1_gaussian_rand(gaussian_std) * rot_factor if random.random() <= 0.6 else 0
    shift = [get_m1to1_gaussian_rand(gaussian_std), get_m1to1_gaussian_rand(gaussian_std)]

    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, shift, color_scale, do_flip



def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape, shift_wh=None, mask=None):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
        if mask is not None:
            mask = mask[:, ::-1]

    trans, shift_xy = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, return_shift=True)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    if mask is not None:
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = (mask > 150).astype(np.uint8)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, inv=True)

    return img_patch, trans, inv_trans, mask, shift_xy

def augmentation(img, bbox, data_split, exclude_flip=False, input_img_shape=(256, 256), mask=None, base_scale=1.1, scale_factor=0.25, rot_factor=60, shift_wh=None, gaussian_std=1, color_aug=False):
    if data_split == 'train':
        scale, rot, shift, color_scale, do_flip = get_aug_config(exclude_flip, base_scale=base_scale, scale_factor=scale_factor, rot_factor=rot_factor, gaussian_std=gaussian_std)
    else:
        scale, rot, shift, color_scale, do_flip = base_scale, 0.0, [0, 0], np.array([1, 1, 1]), False
    do_flip = False
    img, trans, inv_trans, mask, shift_xy = generate_patch_image(img, bbox, scale, rot, shift, do_flip, input_img_shape, shift_wh=shift_wh, mask=mask)
    if color_aug:
        img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, np.array([rot, scale, *shift_xy]), do_flip, input_img_shape[0]/(bbox[3]*scale), mask

def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

class CompHand(data.Dataset):
    def __init__(self, device):
        super(CompHand, self).__init__()
        
        path_lib = Path(os.path.join('/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'))
        # path_lib = Path(os.path.join('/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/base_pose/'))
        
        self.img_list = sorted(list(path_lib.glob('**/pic256/**/*.png')))

        # img_root = '/home/xxx/DGPU/dellnas/datasets/Hand-datasets/MobRecon_Complement_data/Compdata/'
        # filename_lists = getFileList(img_root,'.png')
        # self.imglists=[]
        self.size=128
        self.device = device #torch.device('cuda')
    def __getitem__(self, idx):
        return self.get_contrastive_sample(idx)

    def get_contrastive_sample(self, idx):
        img_path = self.img_list[idx]
        img_name = img_path.parts[-1]
        img_name_split = img_name.split('.')
        num = int(img_name_split[1])

        pose_path = os.path.join(*img_path.parts[:-2], str(num) + '.json').replace('pic256', 'mano_pose')
        mesh_path = os.path.join(*img_path.parts[:-2], str(num) + '.obj').replace('pic256', 'model_mano')
        mask_path = os.path.join(mesh_path.replace('obj', 'png').replace('model_mano', 'mask256'))
        img_path = os.path.join(*img_path.parts)



        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)[..., ::-1, 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        # bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), center[0]+0.5 * max(w, h), center[1]+0.5 * max(w, h)]
        
        # xxx = img.shape[1]*0.7
        # bbox = [0,0,img.shape[1],img.shape[0],1]
        # bbox = [0,0,img.shape[1],img.shape[0],1]
        # ssss=1.5
        # bbox = [center[0]-0.5 * max(w, h)*ssss, center[1]-0.5 * max(w, h)*ssss, center[0]+0.5 * max(w, h)*ssss, center[1]+0.5 * max(w, h)*ssss]

        # # augmentation
        # roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, 'train', exclude_flip=not False, input_img_shape=(128, 128), mask=mask,
        #                                                                              base_scale=1.3, scale_factor=0.2, rot_factor=90,
        #                                                                              shift_wh=[bbox[2], bbox[3]], gaussian_std=3)
        # roi = base_transform(roi, 128)

        # cv2.imshow('vis',img)
        # key = cv2.waitKey(0) & 0xFF

        # breakpoint()
        # cv2.imshow('vis',img[int(bbox[1]):int(bbox[3])+1,int(bbox[0]):int(bbox[2])+1,:])
        # key = cv2.waitKey(0) & 0xFF


        center, scale = box_to_center_scale(bbox, self.size, self.size,scale_factor=1.35)#1.23
        trans = get_affine_transform(center, scale, rot=0, output_size=(self.size,self.size))
        img = cv2.warpAffine(
            np.array(img), trans,
            (int(self.size), int(self.size)),
            flags=cv2.INTER_LINEAR)

        img = img[:, ::-1, ::-1]
        # img = img[:, ::-1, ::]
        img = cv2.resize(img, (self.size, self.size))

        # import cv2
        
        img = torch.from_numpy(base_transform(img, size=self.size)).to(self.device)

        # roi = torch.from_numpy(roi).float()

        # breakpoint()


        # img = img.copy()

        with open(pose_path,'r') as f:
            params = json.load(f)
        # breakpoint()
        gt_pose = np.array(params['poses']).reshape((1,-1))
        gt_Rh = np.array(params['Rh']).reshape((1,-1))
        gt_pose = np.concatenate((gt_Rh,gt_pose),1)

        res = {'img': img, 'gt_params': params,'poses':gt_pose.reshape(48),'path':img_path}
        return res

    def __len__(self):
        return len(self.img_list)




class FreiHAND(data.Dataset):

    def __init__(self, device):
        """Init a FreiHAND Dataset

        Args:
            cfg : config file
            phase (str, optional): train or eval. Defaults to 'train'.
            writer (optional): log file. Defaults to None.
        """
        super(FreiHAND, self).__init__()
        # self.cfg = cfg
        self.size = 128
        self.device = device
        self.phase = 'train'
        self.root = '/home/xxx/DGPU/dellnas/users/wangjunhao/data/FreiHAND'
        self.db_data_anno = tuple(load_db_annotation(self.root, set_name=self.phase))
        # self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None
        self.one_version_len = len(self.db_data_anno)
        if 'train' in self.phase:
            self.db_data_anno *= 4
        # if writer is not None:
        #     writer.print_str('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))))
        # cprint('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def __getitem__(self, idx):
        return self.get_contrastive_sample(idx)


    def get_contrastive_sample(self, idx):


        base_path = self.root
        img_rgb_path = os.path.join(base_path, 'training', 'rgb', '%08d.jpg' % idx)
        img = cv2.imread(img_rgb_path)
        pose_idx = idx % self.one_version_len
        pose_path = os.path.join(base_path, 'training', 'mano_pose',
                             '%08d.json' % pose_idx)
        """Get contrastive FreiHAND samples for consistency learning
        """
        # read
        # img = read_img_abs(idx, self.cfg.DATA.FREIHAND.ROOT, 'training')
        # vert = read_mesh(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT).x.numpy()
        mask = read_mask_woclip(idx % self.one_version_len, self.root, 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        # bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), center[0]+0.5 * max(w, h), center[1]+0.5 * max(w, h)]


        # breakpoint()
        # cv2.imshow('vis',img[int(bbox[1]):int(bbox[3])+1,int(bbox[0]):int(bbox[2])+1,:])
        # key = cv2.waitKey(0) & 0xFF

        center, scale = box_to_center_scale(bbox, self.size, self.size,scale_factor=1.35)#1.23
        trans = get_affine_transform(center, scale, rot=0, output_size=(self.size,self.size))
        img = cv2.warpAffine(
            np.array(img), trans,
            (int(self.size), int(self.size)),
            flags=cv2.INTER_LINEAR)

        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.size, self.size))

        # import cv2
        
        img = torch.from_numpy(base_transform(img, size=self.size)).to(self.device)

        # roi = torch.from_numpy(roi).float()

        # breakpoint()


        # img = img.copy()

        with open(pose_path,'r') as f:
            params = json.load(f)
        # breakpoint()
        gt_pose = np.array(params['poses']).reshape((1,-1))
        gt_Rh = np.array(params['Rh']).reshape((1,-1))
        gt_pose = np.concatenate((gt_Rh,gt_pose),1)

        res = {'img': img, 'gt_params': params,'poses':gt_pose.reshape(48),'path':img_rgb_path}
        return res

        # K, mano, joint_cam = self.db_data_anno[idx]
        # K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        # joint_img = projectPoints(joint_cam, K)
        # princpt = K[0:2, 2].astype(np.float32)
        # focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)
        # # multiple aug
        # roi_list = []
        # calib_list = []
        # mask_list = []
        # vert_list = []
        # joint_cam_list = []
        # joint_img_list = []
        # aug_param_list = []
        # bb2img_trans_list = []
        # for _ in range(2):
        #     # augmentation
        #     roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, roi_mask = augmentation(img.copy(), bbox, self.phase,
        #                                                                                     exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
        #                                                                                     input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
        #                                                                                     mask=mask.copy(),
        #                                                                                     base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
        #                                                                                     scale_factor=self.cfg.DATA.FREIHAND.SCALE,
        #                                                                                     rot_factor=self.cfg.DATA.FREIHAND.ROT,
        #                                                                                     shift_wh=[bbox[2], bbox[3]],
        #                                                                                     gaussian_std=self.cfg.DATA.STD)
        #     if self.color_aug is not None:
        #         roi = self.color_aug(roi)
        #     roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
        #     # img = inv_based_tranmsform(roi)
        #     # cv2.imshow('test', img)
        #     # cv2.waitKey(0)
        #     roi = torch.from_numpy(roi).float()
        #     roi_mask = torch.from_numpy(roi_mask).float()
        #     bb2img_trans = torch.from_numpy(bb2img_trans).float()
        #     aug_param = torch.from_numpy(aug_param).float()

        #     # joints
        #     joint_img_, princpt_ = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
        #     joint_img_ = torch.from_numpy(joint_img_[:, :2]).float() / self.cfg.DATA.SIZE

        #     # 3D rot
        #     rot = aug_param[0].item()
        #     rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
        #                             [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
        #                             [0, 0, 1]], dtype=np.float32)
        #     joint_cam_ = torch.from_numpy(np.dot(rot_aug_mat, joint_cam.T).T).float()
        #     vert_ = torch.from_numpy(np.dot(rot_aug_mat, vert.T).T).float()

        #     # K
        #     focal_ = focal * roi.size(1) / (bbox[2]*aug_param[1])
        #     calib = np.eye(4)
        #     calib[0, 0] = focal_[0]
        #     calib[1, 1] = focal_[1]
        #     calib[:2, 2:3] = princpt_[:, None]
        #     calib = torch.from_numpy(calib).float()

        #     roi_list.append(roi)
        #     mask_list.append(roi_mask.unsqueeze(0))
        #     calib_list.append(calib)
        #     vert_list.append(vert_)
        #     joint_cam_list.append(joint_cam_)
        #     joint_img_list.append(joint_img_)
        #     aug_param_list.append(aug_param)
        #     bb2img_trans_list.append(bb2img_trans)

        # roi = torch.cat(roi_list, 0)
        # mask = torch.cat(mask_list, 0)
        # calib = torch.cat(calib_list, 0)
        # joint_cam = torch.cat(joint_cam_list, -1)
        # vert = torch.cat(vert_list, -1)
        # joint_img = torch.cat(joint_img_list, -1)
        # aug_param = torch.cat(aug_param_list, 0)
        # bb2img_trans = torch.cat(bb2img_trans_list, -1)

        # # postprocess root and joint_cam
        # root = joint_cam[0].clone()
        # joint_cam -= root
        # vert -= root
        # joint_cam /= 0.2
        # vert /= 0.2

        # # out
        # res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask,
        #        'root': root, 'calib': calib, 'aug_param': aug_param, 'bb2img_trans': bb2img_trans,}

        # return res

    # def get_training_sample(self, idx):
    #     """Get a FreiHAND sample for training
    #     """
    #     # read
    #     img = read_img_abs(idx, self.cfg.DATA.FREIHAND.ROOT, 'training')
    #     vert = read_mesh(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT).x.numpy()
    #     mask = read_mask_woclip(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT, 'training')
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = list(contours)
    #     contours.sort(key=cnt_area, reverse=True)
    #     bbox = cv2.boundingRect(contours[0])
    #     center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
    #     w, h = bbox[2], bbox[3]
    #     bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
    #     K, mano, joint_cam = self.db_data_anno[idx]
    #     K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
    #     joint_img = projectPoints(joint_cam, K)
    #     princpt = K[0:2, 2].astype(np.float32)
    #     focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

    #     # augmentation
    #     roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, self.phase,
    #                                                                                     exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
    #                                                                                     input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
    #                                                                                     mask=mask,
    #                                                                                     base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
    #                                                                                     scale_factor=self.cfg.DATA.FREIHAND.SCALE,
    #                                                                                     rot_factor=self.cfg.DATA.FREIHAND.ROT,
    #                                                                                     shift_wh=[bbox[2], bbox[3]],
    #                                                                                     gaussian_std=self.cfg.DATA.STD)
    #     if self.color_aug is not None:
    #         roi = self.color_aug(roi)
    #     roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
    #     # img = inv_based_tranmsform(roi)
    #     # cv2.imshow('test', img)
    #     # cv2.waitKey(0)
    #     roi = torch.from_numpy(roi).float()
    #     mask = torch.from_numpy(mask).float()
    #     bb2img_trans = torch.from_numpy(bb2img_trans).float()

    #     # joints
    #     joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
    #     joint_img = torch.from_numpy(joint_img[:, :2]).float() / self.cfg.DATA.SIZE

    #     # 3D rot
    #     rot = aug_param[0]
    #     rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
    #                             [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    #                             [0, 0, 1]], dtype=np.float32)
    #     joint_cam = np.dot(rot_aug_mat, joint_cam.T).T
    #     vert = np.dot(rot_aug_mat, vert.T).T

    #     # K
    #     focal = focal * roi.size(1) / (bbox[2]*aug_param[1])
    #     calib = np.eye(4)
    #     calib[0, 0] = focal[0]
    #     calib[1, 1] = focal[1]
    #     calib[:2, 2:3] = princpt[:, None]
    #     calib = torch.from_numpy(calib).float()

    #     # postprocess root and joint_cam
    #     root = joint_cam[0].copy()
    #     joint_cam -= root
    #     vert -= root
    #     joint_cam /= 0.2
    #     vert /= 0.2
    #     root = torch.from_numpy(root).float()
    #     joint_cam = torch.from_numpy(joint_cam).float()
    #     vert = torch.from_numpy(vert).float()

    #     # out
    #     res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask, 'root': root, 'calib': calib}

    #     return res

    # def get_eval_sample(self, idx):
    #     """Get FreiHAND sample for evaluation
    #     """
    #     # read
    #     img = read_img(idx, self.cfg.DATA.FREIHAND.ROOT, 'evaluation', 'gs')
    #     K, scale = self.db_data_anno[idx]
    #     K = np.array(K)
    #     princpt = K[0:2, 2].astype(np.float32)
    #     focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)
    #     bbox = [img.shape[1]//2-50, img.shape[0]//2-50, 100, 100]
    #     center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
    #     w, h = bbox[2], bbox[3]
    #     bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]

    #     # aug
    #     roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, _ = augmentation(img, bbox, self.phase,
    #                                                                                     exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
    #                                                                                     input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
    #                                                                                     mask=None,
    #                                                                                     base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
    #                                                                                     scale_factor=self.cfg.DATA.FREIHAND.SCALE,
    #                                                                                     rot_factor=self.cfg.DATA.FREIHAND.ROT,
    #                                                                                     shift_wh=[bbox[2], bbox[3]],
    #                                                                                     gaussian_std=self.cfg.DATA.STD)
    #     roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
    #     roi = torch.from_numpy(roi).float()

    #     # K
    #     focal = focal * roi.size(1) / (bbox[2]*aug_param[1])
    #     calib = np.eye(4)
    #     calib[0, 0] = focal[0]
    #     calib[1, 1] = focal[1]
    #     calib[:2, 2:3] = princpt[:, None]
    #     calib = torch.from_numpy(calib).float()

    #     return {'img': roi, 'calib': calib}

    def __len__(self):

        return len(self.db_data_anno)
