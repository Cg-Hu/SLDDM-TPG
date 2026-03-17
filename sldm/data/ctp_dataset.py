# coding=utf-8
import os

import PIL
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os, os.path as osp
from PIL import Image, ImageDraw
import json
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CenterCropAndResizeOrPad(object):
    def __init__(self, crop_size, center_resize_prob=-1, direct_resize_prob=0.5):

        pass
    def __call__(self, cloth_img):
        # 直接resize cloth图像，与baseline的做法一致
        pass


def mask2bbox(mask):
    # assert mask.size != 0, "debug for cm mask"
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)

class CTPDataset(data.Dataset):
    """
        Dataset for CP-VTON.
    """

    def __init__(self, dataroot, image_size=512, mode='train', semantic_nc=13):
        super(CTPDataset, self).__init__()
        self.data_path = dataroot
        self.datamode = mode  # train or test or self-defined
        self.data_list = mode + '.txt'

        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 256)
        self.semantic_nc = semantic_nc
        
        
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))
        
        # self.cloth_aug = transforms.Compose([
        #     # 亮度、对比度、饱和度和色调
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
        #     ], p=0.8),
        #     # 模糊
        #     transforms.RandomApply([GaussianBlur([.1, .5])], p=0.5), 
        #     # 镜像翻转
        #     transforms.RandomHorizontalFlip(),
        #     # 正方形衣服center后再处理
        # ])

        im_names = [] 
        
        with open(osp.join(self.data_path, self.data_list), 'r') as f:
            for line in f.readlines():
                # im_name, c_name = line.strip().split()
                im_name = line.strip()
                im_names.append(im_name)
            f.close()
        im_names = sorted(im_names)
        self.im_names = im_names
        # self.len_datasets = len(self.im_names)

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        # result = self.getTwo(2*index % self.len_datasets)
        # temp = self.getTwo((2*index+1) % self.len_datasets)
        # result['GT'].append(temp['GT'][0])
        # result['inpaint_image'].append(temp['inpaint_image'][0])
        # result['inpaint_mask'].append(temp['inpaint_mask'][0])
        # result['ref_imgs'].append(temp['ref_imgs'][0])
        # result['file_name'].append(temp['file_name'][0])
        result = self.getTwo(index)
        return result
        
    
    def getTwo(self, index):
        im_name = self.im_names[index]
        cloth_path = os.path.join(self.data_path, 'cloth', im_name)
        if not os.path.exists(cloth_path):
            cloth_path = os.path.join(self.data_path, 'cloth', im_name.split('.')[0]+'.jpg')

        cloth_mask_path = os.path.join(self.data_path, 'cloth_mask', im_name)
        if not os.path.exists(cloth_mask_path):
            cloth_mask_path = os.path.join(self.data_path, 'cloth_mask', im_name.split('.')[0]+'.jpg')

        pattern_path = os.path.join(self.data_path, 'pattern', im_name)
        if not os.path.exists(pattern_path):
            pattern_path = os.path.join(self.data_path, 'pattern', im_name.split('.')[0]+'.jpg')


        cloth_img_pil = Image.open(cloth_path).convert('RGB')
        cloth_img = transforms.Resize(self.crop_size, interpolation=3)(cloth_img_pil)
        
        # ref_cloth_img = self.transform(cloth_img) # 专门供给maskbox裁剪ref图像

        # cloth_img = self.cloth_aug(cloth_img)
        cloth_img = self.transform(cloth_img)  # [-1,1]

        cloth_mask_img_pil = Image.open(cloth_mask_path).convert('1') # 这个mask应该是在人身上的mask 而且是未warp的
        cloth_mask_img = transforms.Resize(self.crop_size, interpolation=0)(cloth_mask_img_pil)
        cm_array = np.array(cloth_mask_img)
        cm_array = cm_array.astype(int)
        # cm_array = (cm_array >= 128).astype(np.float32) # 得到衣服的mask转为二值，0为背景，1为衣服mask
        cloth_mask_img = torch.from_numpy(cm_array)  # [0,1]
        cloth_mask_img.unsqueeze_(0)
        
        pattern_img = Image.open(pattern_path).convert('RGB')
        pattern_img = transforms.Resize(self.crop_size, interpolation=3)(pattern_img)
        # pattern_img = self.cloth_aug(pattern_img)
        pattern_img = self.transform(pattern_img)

        
        # 下面这个代码单纯是把衣服扣出来不要多余的黑色部分
        mask_numpy = cloth_mask_img[0].numpy()
        assert np.sum(mask_numpy) > 0, "This Cloth Is All Black"
        down, up, left, right = mask2bbox(mask_numpy)
        assert up-down > 0 or right-left > 0, f"The Ref Img Is None Or Error"
        ref_image = cloth_img[:, down:up, left:right] # numpy(H,W,C) tensor(C,H,W)
        # cloth_img_center = ref_image
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = transforms.Resize((224, 224))(ref_image)
        ref_image = self.clip_normalize(ref_image) # ref_img应该就是CLIP要用到的参考图

        # if(abs(abs(up-down)-abs(right-left))) <= 50 or random.random() < 0.4: # 如果衣服正方形最好裁剪后再resize
        #     cloth_img = transforms.Resize(self.crop_size, interpolation=3)(cloth_img_center)


        # feat = cloth_img

        # TODO 检查是否提取到的一幅图片是否只是包含衣服 OK

        # TODO 从pattern模拟出逼真的cloth图形，并以此作为cloth_img

        result = {
            "GT": pattern_img, # pattern
            "inpaint_image": cloth_img, # 衣服图
            "inpaint_mask": cloth_mask_img, # 需要修复的部分，cloth部分
            "ref_imgs": ref_image, # 衣服图（中心截取为224x224）而已
            "file_name": self.im_names[index]
        }
        return result

    def __len__(self):
        return len(self.im_names)


if __name__ == '__main__':
    save_path = '/nfs5/hcg/datasets/temp'
    dataset = CTPDataset('/nfs5/hcg/datasets/VITON-HD-SMALL', 512, mode='train')
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    for data in loader:
        cloth = data['inpaint_image']
        name = data['file_name']
        cloth = cloth.squeeze(0)
        print(cloth[0].shape)
        print(cloth[1].shape)
        to_pil = transforms.ToPILImage()
        image = to_pil(cloth[0])

        # 保存为 RGB 图像
        image.save(osp.join(save_path, name[0]))

# /nfs5/hcg/datasets/VITON-HD-SMALL/train.txt