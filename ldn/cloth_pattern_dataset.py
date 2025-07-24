# coding=utf-8
import os, os.path as osp

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw

import json
import random
import numpy as np
from tqdm import tqdm

from data_aug import CenterCropAndResizeOrPad, PatternToCloth, GaussianBlur


class ClothPatternDataset(data.Dataset):
    """
        Dataset for Cloth Pattern Represenation(First Stage).
        mode: train, test, representaion
    """

    def __init__(self, dataroot, image_size=512, mode='train'):
        super(ClothPatternDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.ref_mask_path = osp.join(dataroot, 'ref_mask')
        self.data_list = mode + '.txt'
        self.fine_height = image_size
        self.fine_width = image_size
        self.data_path = dataroot
        
        self.crop_size = (self.fine_height, self.fine_width)

        # 对cloth图像和pattern图像的操作
        augmentation = [
            # cloth的中心裁剪黑边 or 直接resize
            CenterCropAndResizeOrPad(self.crop_size, center_resize_prob=0.3, direct_resize_prob=0.7),
            # 亮度
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            # 灰度
            # transforms.RandomGrayscale(p=0.2),
            # 高斯模糊
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # 翻转
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))
            # TODO 待计算Cloth图像mean和std                                       
        ]
        pattern_aug = [
            transforms.Resize(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))]

        # augmentation_pattern = [
        #     transforms.Resize(self.crop_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     # TODO 待计算Pattern图像mean和std
        # ]

        augmentation_p2c = [
            # 尺度大小+局部不全（概率）
            PatternToCloth(self.ref_mask_path, scale_factor = [0.35, 0.5], not_scale = 0.2),
            # 清晰度
            # transforms.RandomApply([GaussianBlur([.1, .5])], p=0.5), 
            # TODO 扭曲+褶皱
                               
        ]

        self.transforms= transforms.Compose(augmentation)
        # self.transforms_pattern = transforms.Compose(augmentation_pattern)
        self.transforms_p2c = transforms.Compose(augmentation_p2c)
        # self.pattern_aug = transforms.Compose(pattern_aug)


        im_names = []
        if mode in ['train', 'test']:
            print('train and test')
            with open(osp.join(self.data_path, self.data_list), 'r') as f:
                for line in f.readlines():
                    im_name = line.strip()
                    im_names.append(im_name)
                f.close()
        else:
            print('representation')
            im_names = sorted(os.listdir(osp.join(self.data_path, 'cloth')))
            # im_names = os.listdir(self.data_path, 'cloth')
 

        self.im_names = im_names

    def name(self):
        return "Cloth_Pattern_Dataset"


    def __getitem__(self, index):
        im_name = self.im_names[index]
        cloth_path = osp.join(self.data_path, 'cloth', im_name)
        # pattern_path = osp.join(self.data_path, 'pattern_left_crop', im_name.split('.')[0]+'.jpg')
        pattern_path = osp.join(self.data_path, 'pattern', im_name.split('.')[0]+'.jpg')

        cloth_img_tensor = self.transforms(Image.open(cloth_path).convert('RGB'))

        pattern_img_pil = Image.open(pattern_path).convert('RGB')

        pattern_img_tensor = self.transforms(pattern_img_pil)

        p2c_img_tensor = self.transforms(self.transforms_p2c(pattern_img_pil))

        # 下一步训练使用
        # cloth_mask_path = osp.join(self.data_path, 'cloth_mask', im_name)
        # cloth_mask_img_pil = Image.open(cloth_mask_path)
        # cloth_mask_img_pil = transforms.Resize(self.crop_size, interpolation=0)(cloth_mask_img_pil)
        # cloth_mask_img_array = np.array(cloth_mask_img_pil)
        # cloth_mask_img_array = (cloth_mask_img_array >= 128).astype(np.float32) # 得到衣服的mask转为二值，0为背景，1为衣服mask
        # cloth_mask_img_tensor = torch.from_numpy(cloth_mask_img_array)  # [0,1]
        # cloth_mask_img_tensor.unsqueeze_(0)

        result = {
            "im_name": im_name,
            "pattern": pattern_img_tensor, # pattern
            "cloth": cloth_img_tensor, # cloth
            "p2c": p2c_img_tensor, # pattern to cloth aug 模拟
            # "cloth_mask": cloth_mask_img_tensor
        }
        return result

    def __len__(self):
        return len(self.im_names)
    

    # TODO 检测一下数据集的效果，需要保存一些图片看看
    # v1:暂定数据质量不高，得分析原因
    # v2:去除了中心裁剪后resize，加入直接resize cloth_img，去除了pattern的左边无关像素，现在效果还行 
    #    TODO 多加入一些ref_mask图像

if __name__ == '__main__':
    save_root = '/nfs5/hcg/datasets/cloth_pattern'
    dataset = ClothPatternDataset('/nfs5/hcg/datasets/VITON-HD-V2', 512, mode='representation')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for data in tqdm(loader, desc='load dataset'):
        pattern, cloth, p2c = data['pattern'], data['cloth'], data['p2c']
        img_name = data['im_name'][0]
        to_pil = transforms.ToPILImage()
        pattern_pil, cloth_pil, p2c_pil = to_pil(pattern.squeeze(0)), to_pil(cloth.squeeze(0)), to_pil(p2c.squeeze(0))
        pattern_pil.save(osp.join(save_root, img_name+'_pattern.png'))
        cloth_pil.save(osp.join(save_root, img_name+'_cloth.png'))
        p2c_pil.save(osp.join(save_root, img_name+'_p2c.png'))
