import random
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import torch
from torchvision.transforms import transforms
import simsiam.loader
import os

__all__ = ['CenterCropAndResizeOrPad', 'PatternToCloth', 'GaussianBlur']

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
        self.w, self.h = crop_size[0], crop_size[1]
        self.center_resize_prob = center_resize_prob
        self.direct_resize_prob = direct_resize_prob

    def __call__(self, cloth_img):
        # 直接resize cloth图像，与baseline的做法一致
        if random.random() < self.direct_resize_prob:
            cloth_img = cloth_img.resize((self.w, self.h), Image.LANCZOS)
            return cloth_img
        else:
            # Convert the input PIL image to numpy array for easier manipulation
            cloth_img_array = np.array(cloth_img)
            
            # Find the bounding box of the non-black area
            non_black_pixels = np.argwhere(cloth_img_array > 0)
            if non_black_pixels.size == 0:
                raise ValueError("Image has no content to crop.")
            
            cloth_mask = np.where((cloth_img_array[..., :3] == [0, 0, 0]).all(axis=-1), 0, 1).astype(np.uint8)
            up = np.max(np.where(cloth_mask)[0])
            down = np.min(np.where(cloth_mask)[0])
            left = np.min(np.where(cloth_mask)[1])
            right = np.max(np.where(cloth_mask)[1])
            # center = ((up + down) // 2, (left + right) // 2)
            # factor = random.random() * 0.1 + 0.1
            # up = int(min(up * (1 + factor) - center[0] * factor + 1, cloth_mask.shape[0]))
            # down = int(max(down * (1 + factor) - center[0] * factor, 0))
            # left = int(max(left * (1 + factor) - center[1] * factor, 0))
            # right = int(min(right * (1 + factor) - center[1] * factor + 1, cloth_mask.shape[1]))
            center_cloth_img_array =cloth_img_array[down:up, left:right, :]
            cropped_width, cropped_height, _ = center_cloth_img_array.shape
            center_cloth_img_pil = Image.fromarray(center_cloth_img_array)
            
            if random.random() < self.center_resize_prob:
                # Resize the image to the target size
                resized_center_cloth_img_pil = center_cloth_img_pil.resize((self.w, self.h), Image.LANCZOS)

            else:
                # Padding to target size with black pixels
                pad_left = 0
                pad_right = 0
                pad_bottom = 0
                pad_top = 0
                if cropped_height < self.h - 1:
                    pad_left = (self.h - cropped_height) // 2
                    pad_right = self.h - cropped_height - pad_left
                if cropped_width < self.w - 1:
                    pad_top = (self.w - cropped_width) // 2
                    pad_bottom = self.w - cropped_width - pad_top
                resized_center_cloth_img_pil = ImageOps.expand(center_cloth_img_pil, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
                if resized_center_cloth_img_pil.size != (self.w, self.h):
                    resized_center_cloth_img_pil = resized_center_cloth_img_pil.resize((self.w, self.h), Image.LANCZOS)
        return resized_center_cloth_img_pil


class PatternToCloth(object):
    def __init__(self, cloth_mask_root, scale_factor=[0.35, 0.5], not_scale = 0.2) -> None:
        cloth_mask_names = os.listdir(cloth_mask_root)
        self.cloth_mask_path = [os.path.join(cloth_mask_root, name) for name in cloth_mask_names]
        self.scale_factor = scale_factor
        self.not_scale = not_scale
    
    def __call__(self, pattern_img):
        mask_img = Image.open(self.cloth_mask_path[random.choice(range(len(self.cloth_mask_path)))] )  
        # 决定是否进行缩放
        if random.random() > self.not_scale:
            
            sf = random.uniform(self.scale_factor[0], self.scale_factor[1])
            small_w, small_h = int(pattern_img.width * sf), int(pattern_img.height * sf)
            small_pattern = pattern_img.resize((small_w, small_h), Image.LANCZOS)
            output_size=mask_img.size
            # 创建一个空白的输出图像
            tiled_pattern_img = Image.new('RGB', output_size)
            
            # 平铺小图像，填满整个输出图像
            for i in range(0, output_size[0], small_w):
                for j in range(0, output_size[1], small_h):
                    tiled_pattern_img.paste(small_pattern, (i, j))
            masked_result = Image.composite(tiled_pattern_img, Image.new('RGB', mask_img.size, (0, 0, 0)), mask_img)
            return masked_result
        else:
            tiled_pattern_img = pattern_img
            mask_img = mask_img.resize((pattern_img.size[0], pattern_img.size[1]), Image.LANCZOS)
            masked_result = Image.composite(tiled_pattern_img, Image.new('RGB', mask_img.size, (0, 0, 0)), mask_img)
            return masked_result
        


        
     
def process_pattern_image(img):
    """
    Resize the pattern image and tile it to 512x512, then apply a distorted rectangle mask.
    """
    # Convert image to numpy array
    img_array = np.array(img)

    # Resize pattern image
    scale_factor = 0.5
    new_width = int(img_array.shape[1] * scale_factor)
    new_height = int(img_array.shape[0] * scale_factor)
    pattern_img = Image.fromarray(img_array).resize((new_width, new_height), Image.LANCZOS)

    # Tile pattern to 512x512
    tiled_img = Image.fromarray(np.tile(np.array(pattern_img), (512 // new_height + 1, 512 // new_width + 1, 1))[:512, :512, :])

    # Create a long skirt-like mask in the center of the image
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    top_width = random.randint(200, 300)
    bottom_width = random.randint(400, 500)
    height = 512
    points = [
        (256 - top_width // 2, 0),
        (256 + top_width // 2, 0),
        (256 + bottom_width // 2, height),
        (256 - bottom_width // 2, height)
    ]
    draw.polygon(points, fill=255)

    # Apply mask to tiled image
    masked_img = Image.composite(tiled_img, Image.new('RGB', (), (0, 0, 0)), mask)

    return masked_img


def tile_pattern_image(pattern_img, mask_img, scale_factor=random.uniform(0.35, 0.5)): # 局部不全
    # 缩小 pattern 图像
    small_w, small_h = int(pattern_img.width * scale_factor), int(pattern_img.height * scale_factor)
    small_pattern = pattern_img.resize((small_w, small_h), Image.LANCZOS)
    output_size=mask_img.size
    # 创建一个空白的输出图像
    tiled_pattern_img = Image.new('RGB', output_size)
    
    # 平铺小图像，填满整个输出图像
    for i in range(0, output_size[0], small_w):
        for j in range(0, output_size[1], small_h):
            tiled_pattern_img.paste(small_pattern, (i, j))
    
    return tile_pattern_image
    # mask_3d = Image.merge('RGB', [mask_img, mask_img, mask_img])

    # 使用 mask 进行遮罩操作
    masked_result = Image.composite(tiled_pattern_img, Image.new('RGB', mask_img.size, (0, 0, 0)), mask_img)
    return masked_result
    
if __name__ == '__main__':

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation_cloth = [
        # 亮度
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # not strengthened
        ], p=0.8),
        # cloth的中心裁剪黑边 or resize
        CenterCropAndResizeOrPad(crop_size=(512, 512), center_resize_prob=0.5),  # Add the new custom transform here
        transforms.ToTensor(),
        normalize
    ]

    augmentation_pattern = [
        transforms.Resize((512, 512)),
        # transforms.RandomApply([simsiam.loader.GaussianBlur([0.1, 1])], p=0.5),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_c2p = [
        # 尺度大小+局部不全（概率）
        PatternToCloth('/nfs5/hcg/datasets/VITON-HD-SMALL/test/cloth_mask', scale_factor = [0.35, 0.5], not_scale = 0.2),
        # 清晰度
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, .5])], p=0.5), 
        # 扭曲+褶皱
    ]

    # transform = transforms.Compose(augmentation_pattern)
    transform = transforms.Compose(augmentation_c2p)
    img_root = '/nfs5/hcg/datasets/VITON-HD-SMALL/train/pattern'
    img_files = os.listdir(img_root)
    for img_name in img_files:
        img_path = os.path.join(img_root, img_name)
        img = Image.open(img_path)
        mask_img = Image.open(img_path.replace('pattern', 'cloth_mask').replace('jpg', 'png'))
        # img = process_pattern_image(img)
        # img = tile_pattern_image(img)
        
        aug_img = transform(img)
        aug_img.save(f"{img_name.split('.')[0]}_c2p.png")
        # print(type(aug_img))