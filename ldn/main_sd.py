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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import heapq  # 用于追踪最佳权重
from cloth_pattern_dataset import ClothPatternDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from datetime import datetime, timedelta
from ldn import ClothPatternSimDiff
from utils import WarmupCosineScheduler, dataloadAndaug
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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, # 32
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-acc', '--accumulate_steps', default=16, type=int, # 8
                    metavar='N',
                    help='accumulate gradient')
parser.add_argument('--lr', '--learning-rate', default=5e-2, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--flr', '--final-lr', default=5e-4, type=float,
                    metavar='FLR', help='initial (base) final learning rate', dest='flr')
parser.add_argument('--alpha_margin', default=0.2, type=float,
                    metavar='A', help='Triplet loss margin')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default=f"", type=str, metavar='PATH',
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
                    help='hidden dimension of the predictor (default: 1024)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--scm-epochs', default=100, type=int,
                    help='Train For Sim')


# auto_encoder specific configs:
parser.add_argument(
        "--config",
        type=str,
        default=f"/home/hcg/cloth_pattern/SLDDM-TPG/ldn/first_autoencoder/encoder.yaml",
        help="path to config which constructs model",
)

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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

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
            args.rank = gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # logs tensorboard
    writer = None
    root = None
    
    if args.gpu == 0:
        time_delta = timedelta(hours=8)

        now = (datetime.now() + time_delta).strftime("%Y-%m-%dT%H:%M:%S") + "ckpt_epoch87"
        root = os.path.join(f"logs", now)
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
    model = ClothPatternSimDiff(layers = [3, 4, 6, 3], config = args.config, feature_dim=2048, pred_dim=1024, recover_dim=4096, feature_mode='res', first_train=True, scm_epochs=args.scm_epochs)
    

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion_cos = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    criterion_mse = nn.MSELoss(reduction='mean').cuda(args.gpu)
    criterion = [criterion_cos, criterion_mse]
    # datasets
    train_loader, train_sampler = dataloadAndaug(args=args, mode='train')
    test_loader, test_sampler = dataloadAndaug(args=args, mode='test')
    epoch_steps = len(train_loader.dataset) // (args.accumulate_steps * args.batch_size)  # 训练数据集的大小
    

    # params adjust
    # init_lr = args.lr * args.batch_size  * 2 * args.accumulate_steps / 256
    # init_flr = args.flr * args.batch_size * 2 * args.accumulate_steps / 256
    init_lr = args.lr
    init_flr = args.flr

    if args.fix_pred_lr:
        print(f"====> Fix the predictor learning rate")
        optim_params = [{'params': model.module.feature_encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.sim.fc_sim.parameters(), 'fix_lr': False},
                        {'params': model.module.sim.predictor.parameters(), 'fix_lr': True},
                        {'params': model.module.diff.parameters(), 'fix_lr': False},
                        {'params': model.module.affine_block.parameters(), 'fix_lr': False}]
    else:
        optim_params = model.parameters()
    
    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=init_lr,
                                 weight_decay=args.weight_decay, # clip is 0.2
                                 betas=(0.9, 0.98),
                                 eps=1e-6)
    
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=args.epochs, min_lr=init_flr)

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
            # model.load_state_dict(checkpoint, strict=False)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    max_epoch_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        avg_train_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args, writer, epoch_steps)
        if args.distributed:
            test_sampler.set_epoch(epoch)
        avg_val_loss = test(test_loader, model, criterion, epoch, args, writer)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
            if avg_val_loss < max_epoch_loss:
                is_best = True
                max_epoch_loss = avg_val_loss

            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, save_path=save_path, epoch=epoch, best_path=best_path, avg=avg_val_loss)


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args, writer, epoch_steps):
    criterion_cos = criterion[0]
    criterion_mse = criterion[1]
    # switch to train mode
    model.train()
    accumulate_steps = args.accumulate_steps
    optimizer.zero_grad()
    total_loss = 0.0  # Track loss for logging purposes
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training Epoch {epoch}', leave=True, dynamic_ncols=True, ncols=100)
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
        # outputs = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)
        if epoch > args.scm_epochs:
            p1, p2, p_aug, z1, z2, z_aug, x1_diff, x2_diff, xaug_diff, x2_affine, xaug_affine = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)
        else:
            p1, p2, p_aug, z1, z2, z_aug = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)

        # 1 SCM Module
        pc = criterion_cos(p1, z2).mean()
        cp = criterion_cos(p2, z1).mean()
        pc_aug = criterion_cos(p1, z_aug).mean()
        c_augp = criterion_cos(p_aug, z1).mean()
        pc_loss = -(pc + cp) * 0.5  # the lowest loss is -1
        pp_loss = -(pc_aug + c_augp) * 0.001 # the lowest loss -0.002
        sim_loss = pc_loss + pp_loss # the lowest loss -1.002
        if epoch > args.scm_epochs:
            # 2 RAM Module
            # 2.1 Diff
            pcd = criterion_mse(x1_diff, x2_diff).mean()
            pcaugd = criterion_mse(x1_diff, xaug_diff).mean()
            #2.2 SATs 
            pc_affine = criterion_mse(x1_diff, x2_affine).mean()
            pcaug_affine = criterion_mse(x1_diff, xaug_affine).mean()
            gap_loss = (pcd + 0.001 * pcaugd) * 0.5 # 远离 # -0.5005
            affine_loss = -(pc_affine + 0.001 * pcaug_affine) * 0.2 # 仿射后需要接近(这个就有点像对抗学习了) # -0.2002
            cotrain_loss = gap_loss + affine_loss + args.alpha_margin # the lowest loss -0.7007
            loss = sim_loss + cotrain_loss # -1.002 - 0.7007 = -1.7027
        else:
            loss = sim_loss # -1.2 
        # loss = loss / accumulate_steps  # 平摊损失 # -0.075 - 0.048125 = 0.123125
        (loss / accumulate_steps).backward()

        progress_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()

        
        # 每 accumulate_steps 个 batch 更新一次权重
        if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            lr_scheduler.step(epoch + float(i) / float(epoch_steps))  # 更新学习率
            optimizer.zero_grad()
                

        if writer is not None:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train_LR/lr', lr_scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Train_Loss/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train_Loss/Sim_Loss', sim_loss.item(), global_step)
            writer.add_scalar('Train_Loss/PC_Loss', pc_loss.item(), global_step)
            writer.add_scalar('Train_Loss/PP_Loss', pp_loss.item(), global_step)
            if epoch > args.scm_epochs:
                writer.add_scalar('Train_Loss/Diff_Loss', cotrain_loss.item(), global_step)
                writer.add_scalar('Train_Loss/Gap_Loss', gap_loss.item(), global_step)
                writer.add_scalar('Train_Loss/Affine_Loss', affine_loss.item(), global_step)
                writer.add_scalars('Train_Diff/Multi', {
                    'PC_Diff': pcd.item(),
                    'PCAUG_Diff': pcaugd.item(),
                    'PC_Affine': pc_affine.item(),
                    'PCAUG_Affine': pcaug_affine.item()
                }, global_step)

            # 将所有值绘制到同一个图中
            writer.add_scalars('Train_Sim/Multi', {
                'PC': pc.item(),
                'CP': cp.item(),
                'PC_Aug': pc_aug.item(),
                'C_AugP': c_augp.item()
            }, global_step)
            

    avg_epoch_loss = total_loss / (len(train_loader) // args.batch_size)
    if writer is not None:
        writer.add_scalar('Train_Epoch_Loss', avg_epoch_loss, epoch)
    return avg_epoch_loss

def test(test_loader, model, criterion, epoch, args, writer):
    criterion_cos = criterion[0]
    criterion_mse = criterion[1]
    total_loss = 0
    model.eval()
    with open('/home/hcg/cloth_pattern/Representation/cpsd/logs_sd/106/2025-01-14T16:01:50/result/test_sim_diff.txt', 'a+') as f:
        with torch.no_grad():
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test', leave=True, dynamic_ncols=True, ncols=100)
            for i, data_dict in progress_bar:
                im_name = data_dict['im_name'][0]
                pattern_img_tensor = data_dict['pattern']
                cloth_img_tensor = data_dict['cloth']
                p2c_img_tensor = data_dict['p2c']
                # p2c_img_tensor = transforms_cloth(transforms_p2c(pattern_img_tensor))

                if args.gpu is not None:
                    pattern_img_tensor = pattern_img_tensor.cuda(args.gpu, non_blocking=True)
                    cloth_img_tensor = cloth_img_tensor.cuda(args.gpu, non_blocking=True)
                    p2c_img_tensor = p2c_img_tensor.cuda(args.gpu, non_blocking=True)

                # compute output and loss
            # outputs = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)
            if epoch > args.scm_epochs:
                p1, p2, p_aug, z1, z2, z_aug, x1_diff, x2_diff, xaug_diff, x2_affine, xaug_affine = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)
            else:
                p1, p2, p_aug, z1, z2, z_aug = model(pattern_img_tensor, cloth_img_tensor, p2c_img_tensor, epoch)

            # 1 SCM Module
            pc = criterion_cos(p1, z2).mean()
            cp = criterion_cos(p2, z1).mean()
            pc_aug = criterion_cos(p1, z_aug).mean()
            c_augp = criterion_cos(p_aug, z1).mean()
            pc_loss = -(pc + cp) * 0.5  # the lowest loss is -1
            pp_loss = -(pc_aug + c_augp) * 0.001 # the lowest loss -0.002
            sim_loss = pc_loss + pp_loss # the lowest loss -1.002
            if epoch > args.scm_epochs:
                # 2 RAM Module
                # 2.1 Diff
                pcd = criterion_mse(x1_diff, x2_diff).mean()
                pcaugd = criterion_mse(x1_diff, xaug_diff).mean()
                #2.2 SATs 
                pc_affine = criterion_mse(x1_diff, x2_affine).mean()
                pcaug_affine = criterion_mse(x1_diff, xaug_affine).mean()
                gap_loss = (pcd + 0.001 * pcaugd) * 0.5 # 远离 # -0.5005
                affine_loss = -(pc_affine + 0.001 * pcaug_affine) * 0.2 # 仿射后需要接近(这个就有点像对抗学习了) # -0.2002
                cotrain_loss = gap_loss + affine_loss + args.alpha_margin # the lowest loss -0.7007
                loss = sim_loss + cotrain_loss # -1.002 - 0.7007 = -1.7027
            else:
                loss = sim_loss # -1.2 
            
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

            f.writelines(f"im_name:{im_name}, pc:{pc}, cp:{cp}, pc_aug:{pc_aug}, c_augp:{c_augp}, pcd:{pcd}, pcaugd:{pcaugd}, pc_affine:{pc_affine}, pcaug_affine:{pcaug_affine}, total_loss:{loss}\n")
            if writer is not None:
                
                global_step = epoch * len(test_loader) + i
                writer.add_scalar('Test_Loss/Total_Loss', loss.item(), global_step)
                writer.add_scalar('Test_Loss/Sim_Loss', sim_loss.item(), global_step)
                writer.add_scalar('Test_Loss/PC_Loss', pc_loss.item(), global_step)
                writer.add_scalar('Test_Loss/PP_Loss', pp_loss.item(), global_step)
                
                if epoch > args.scm_epochs:
                    writer.add_scalar('Test_Loss/Diff_Loss', cotrain_loss.item(), global_step)
                    writer.add_scalar('Test_Loss/Gap_Loss', gap_loss.item(), global_step)
                    writer.add_scalar('Test_Loss/Affine_Loss', affine_loss.item(), global_step)
                    writer.add_scalars('Test_Diff/Multi', {
                        'PC_Diff': pcd.item(), # S 
                        'PCAUG_Diff': pcaugd.item(), # S
                        'PC_Affine': pc_affine.item(), # B
                        'PCAUG_Affine': pcaug_affine.item() # B
                    }, global_step)

                # 将所有值绘制到同一个图中
                writer.add_scalars('Test_Sim/Multi', {
                    'PC': pc.item(), # B
                    'CP': cp.item(), # B
                    'PC_Aug': pc_aug.item(), # B
                    'C_AugP': c_augp.item() # B
                }, global_step)

                

            avg_val_loss = total_loss / (len(test_loader) // args.batch_size)
            if writer is not None:
                writer.add_scalar('Test_Epoch_Loss', avg_val_loss, epoch)
            return avg_val_loss

def save_checkpoint(ckpt, is_best, save_path, epoch, best_path, avg):
    # param_list = ['fc_sim', 'predictor', 'patch_embedding', 'final_ln']
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
