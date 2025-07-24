#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import builtins
import math
import os
import random
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# import simsiam.loader
import heapq  # 用于追踪最佳权重
from ldn import ClothPatternSimDiff
# from data_aug import CenterCropAndResizeOrPad, PatternToCloth
from cloth_pattern_dataset import ClothPatternDataset
# from torch.cuda.amp import GradScaler, autocast
# scaler = GradScaler()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime, timedelta

best_checkpoints = []


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-acc', '--accumulate_steps', default=32, type=int,
                    metavar='N',
                    help='accumulate gradient')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--flr', '--final-lr', default=0.001, type=float,
                    metavar='FLR', help='initial (base) final learning rate', dest='flr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--alpha_margin', default=0.2, type=float,
                    metavar='A', help='Triplet loss margin')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=f"home/hcg/cloth_pattern/Representation/cpsd/logs_sim/210/2025-01-05T22:49:08/best/checkpoint_0020.pth.tar", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=2, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12355', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=1024, type=int,
                    help='hidden dimension of the predictor (default: 1024)', dest='pdim')
parser.add_argument('--recover-dim', default=4096, type=int,
                    help='hidden dimension of the predictor (default: 4096)', dest='rdim')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


# auto_encoder specific configs:
parser.add_argument(
        "--config",
        type=str,
        default=f"/home/hcg/cloth_pattern/SLDDM-TPG/ldn/first_autoencoder/encoder.yaml",
        help="path to config which constructs model",
)


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


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, final_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            warmup_lr = self.base_lr * step / self.warmup_steps
            return [warmup_lr for _ in self.optimizer.param_groups]
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            return [self.final_lr + (self.base_lr - self.final_lr) * cosine_decay for _ in self.optimizer.param_groups]


def dataloadAndaug(args):
    
    train_dataset = ClothPatternDataset(args.data, 512, mode='representation')
    # transforms_p2c, transforms_cloth = train_dataset.transforms_p2c, train_dataset.transforms_cloth
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader, train_sampler



def main():
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print('执行这里')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # logs tensorboard
    writer = None
    root = None
    if args.gpu == 0:
        time_delta = timedelta(hours=8)
        now = (datetime.now() + time_delta).strftime("%Y-%m-%dT%H:%M:%S")
        root = os.path.join(f"home/hcg/cloth_pattern/Representation/cpsd/logs_diff", now)
        os.makedirs(root, exist_ok=True)
        log_dir = os.path.join(root, "tf")
        save_path = os.path.join(root, "ckpt")
        best_path = os.path.join(root, 'best')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(best_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = ClothPatternSimDiff(layers = [3, 4, 6, 3], config = args.config, feature_dim=args.dim, pred_dim=args.pdim, recover_dim=args.rdim, train_sim=False, feature_mode='res')

    # datasets
    train_loader, train_sampler = dataloadAndaug(args=args)
    dataset_size = len(train_loader.dataset)  # 训练数据集的大小

    # params adjust
    init_lr = args.lr * args.batch_size * args.accumulate_steps * 2 / 256
    warmup_steps =  dataset_size // args.batch_size // args.accumulate_steps
    total_steps = args.epochs * warmup_steps
    init_flr = args.flr * args.batch_size * args.accumulate_steps * 2 / 256

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion_cos = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    criterion_mse = nn.MSELoss(reduction='mean').cuda(args.gpu)

    # 冻结 feature_encoder 和 sim 网络的所有参数
    for param in model.module.feature_encoder.parameters():
        param.requires_grad = False
    for param in model.module.sim.parameters():
        param.requires_grad = False


    if args.fix_pred_lr:
        optim_params = [{'params': model.module.diff.parameters(), 'fix_lr': False},
                        {'params': model.module.affine_block.parameters(), 'fix_lr': False}]
    else:
        optim_params = model.parameters()

    # optimizer = torch.optim.SGD(optim_params, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=init_lr,
                                 weight_decay=args.weight_decay, # clip is 0.2
                                 betas=(0.9, 0.98),
                                 eps=1e-6)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, base_lr=init_lr, final_lr=init_flr)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         # args.start_epoch = checkpoint['epoch']

    #         args.start_epoch = 0
    #         model.load_state_dict(checkpoint['state_dict'], strict=True)
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    
    
    max_epoch_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        avg_epoch_loss = train(train_loader, model, criterion_mse, optimizer, scheduler, epoch, args, writer)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if avg_epoch_loss < max_epoch_loss:
                is_best = True
                max_epoch_loss = avg_epoch_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, save_path=save_path, epoch=epoch, best_path=best_path, avg=avg_epoch_loss)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, writer):
    # switch to train mode
    model.train()
    accumulate_steps = args.accumulate_steps
    optimizer.zero_grad()
    total_loss = 0.0  # Track loss for logging purposes
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training Epoch {epoch}', leave=True, dynamic_ncols=True)
    for i, data_dict in progress_bar:
        # im_name = data_dict['im_name']
        pattern_img_tensor = data_dict['pattern']
        cloth_img_tensor = data_dict['cloth']
        p2c_img_tensor = data_dict['p2c']
        # p2c_img_tensor = transforms_cloth(transforms_p2c(pattern_img_tensor))

        if args.gpu is not None:
            pattern_img_tensor = pattern_img_tensor.cuda(args.gpu, non_blocking=True)
            cloth_img_tensor = cloth_img_tensor.cuda(args.gpu, non_blocking=True)
            p2c_img_tensor = p2c_img_tensor.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        z1, z2, z_aug, affine_z2_diff, affine_z_aug_diff = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor)
        
        # Accumulate loss for backpropagation
        # 这边是低级特征的差距要最大，而且cloth图像要与增强过后的图像存咋些许的接近，避免这种差距特征完全偏离原来的轨迹
        pc = criterion(z1, z2).mean()
        pc_aug = criterion(z1, z_aug).mean()
        cc_aug = criterion(z2, z_aug).mean()
        pc_affine = criterion(z1, affine_z2_diff).mean()
        pc_aug_affine = criterion(z1, affine_z_aug_diff).mean()

        diff_loss = pc + pc_aug
        
        # 仿射loss
        affine_loss = -pc_affine - 0.01 * pc_aug_affine
        margin = args.alpha_margin
        loss = diff_loss + 0.5 * affine_loss + margin
        loss = loss / accumulate_steps  # 平摊损失
        loss.backward()

        progress_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()
        # 每 accumulate_steps 个 batch 更新一次权重
        if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if writer is not None:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Loss/Diff_Loss', diff_loss.item(), global_step)
            writer.add_scalar('Loss/Affine_Loss', affine_loss.item(), global_step)

            # 将损失值写入 TensorBoard
            writer.add_scalar('SIM/PC', pc.item(), global_step) # 小
            writer.add_scalar('SIM/PC_Aug', pc_aug.item(), global_step) # 小
            writer.add_scalar('SIM/CC_Aug', cc_aug.item(), global_step) # 大
            writer.add_scalar('SIM/PC_Affine', pc_affine.item(), global_step) # 大
            writer.add_scalar('SIM/PC_AUG_Affine', pc_aug_affine.item(), global_step) # 大

            # 将所有损失值绘制在同一张图上
            writer.add_scalars('SIM/Multi', {
                'PC': pc.item(),
                'PC_Aug': pc_aug.item(),
                'CC_Aug': cc_aug.item(),
                'PC_Affine': pc_affine.item(),
                'PC_AUG_Affine': pc_aug_affine.item()
            }, global_step)
    
    avg_epoch_loss = total_loss / len(train_loader)
    if writer is not None:
        writer.add_scalar('Epoch_Loss', total_loss / len(train_loader), epoch)
    return avg_epoch_loss

    



def save_checkpoint(ckpt, is_best, save_path, epoch, best_path, avg):

    # param_list = ['diff', 'affine_block']
    # filtered_state_dict = {k: v for k, v in ckpt['state_dict'].items() if any(name in k for name in param_list) }
    # ckpt['state_dict'] = filtered_state_dict
    # 每隔 10 个 epoch 保存一次权重
    if epoch % 10 == 0:
        filename = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
        torch.save(ckpt, filename)

    # 保存最佳的两个权重
    if is_best:
        heapq.heappush(best_checkpoints, (ckpt['epoch'], ckpt))
        if len(best_checkpoints) > 2:
            # 删除最差的检查点
            best_checkpoints.pop(0)
        
        # 保存前两个最佳权重到 'best' 目录
        if not os.path.exists(f'{best_path}'):
            os.mkdir(f'{best_path}')
        for i, (_, best_ckpt) in enumerate(best_checkpoints):
            if len(best_checkpoints) == 2 and i == 0: # 确保不会重复保存
                continue
            torch.save(best_ckpt, os.path.join(f'{best_path}', f'ckpt_epoch{epoch}_{avg}.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
