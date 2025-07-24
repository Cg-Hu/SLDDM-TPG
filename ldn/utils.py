import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models
from cloth_pattern_dataset import ClothPatternDataset

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_coefficient = nn.Parameter(torch.tensor(0.0))  # 初始化为0

    def forward(self):
        return 0.5 * torch.sigmoid(self.raw_coefficient)  # 将范围约束在 [0, 0.5]
    

# class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, warmup_steps, total_steps, base_lr, final_lr=0.0, last_epoch=-1, last_step=-1):
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.base_lr = base_lr
#         self.final_lr = final_lr
#         self.last_step = last_step
#         super().__init__(optimizer, last_epoch)
    
#     def get_lr(self):
#         step = self.last_step + 1
#         if step < self.warmup_steps:
#             warmup_lr = self.base_lr * step / self.warmup_steps
#             return [warmup_lr for _ in self.optimizer.param_groups]
#         else:
#             cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
#             return [self.final_lr + (self.base_lr - self.final_lr) * cosine_decay for _ in self.optimizer.param_groups]

#     def step(self, step=None):
#         if step is not None:
#             # 如果提供了step参数，更新last_step为该值
#             self.last_step = step
#         else:
#             # 否则自动递增last_step
#             self.last_step += 1
#         super().step()


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            lr_mult = float(epoch) / float(max(1, self.warmup_epochs))
        else:
            progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs)) # epoch的进度
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [max(self.min_lr, float(base_lr) * lr_mult) for base_lr in self.base_lrs]

def center_cloth(mask):
    assert mask.size != 0, "debug for cm mask"
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


def dataloadAndaug(args, mode):
    
    train_dataset = ClothPatternDataset(args.data, 512, mode=mode)
    # transforms_p2c, transforms_cloth = train_dataset.transforms_p2c, train_dataset.transforms_cloth
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader, train_sampler