import os
import os.path as osp
import shutil
from tqdm import tqdm
import json
from os.path import join
import numpy as np
import cv2



if __name__ == '__main__':
    namelist = sorted(os.listdir('/dellnas/dataset/XiaoMiMocap/220715/action+000600+006000/images/07/'))
    for name in tqdm(namelist):
        img = cv2.imread('/dellnas/dataset/XiaoMiMocap/220715/action+000600+006000/images/07/{}'.format(name))
        cv2.imwrite('/nas/users/wangjunhao/out/crop_img/{}'.format(name),img[310:1250,:,:])
    
    # img = cv2.imread('/dellnas/dataset/XiaoMiMocap/220715/action+000600+006000/images/07/{:06d}.jpg'.format(0))
    # print(img.shape)
    # img = img[310:1250,:,:]
    # cv2.imshow('vis',img)
    
    # k = cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()