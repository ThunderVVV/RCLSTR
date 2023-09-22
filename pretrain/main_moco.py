#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

import moco.loader
import moco.builder

from model import Model
from datasets import ImgDataset, LmdbDataset
from imgaug import augmenters as iaa

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, required=True,
                    help='path to dataset')
parser.add_argument("--model_name", type=str, required=True, help="NRBA")
parser.add_argument("--exp_name", type=str, required=True, help="exp name")
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
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

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


# for OCR
parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
parser.add_argument(
        "--SelfSL_layer", type=str, default="CNNLSTM", help="for SelfSL_layer"
    )
parser.add_argument(
        "--self",
        type=str,
        default="MoCoSeqCLR",
        help="whether to use self-supervised learning |MoCo|MoCoSeqCLR|",
    )
parser.add_argument(
        "--instance_map",
        type=str,
        default="window",
        help="window to instance or all to instance, |window|all|",
    )
parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
parser.add_argument('--light_aug', action='store_true',
                    help='use light data augmentation')
parser.add_argument(
        "--data-format",
        type=str,
        default="jpg",
        help="data format, |jpg|lmdb|",
    )

parser.add_argument('--useTPS', default='', type=str, help='use TPS of the path to TRBA model (default: none)')
parser.add_argument(
        "--loss_setting",
        type=str,
        default="",
        help="loss setting, ||consistent|",
    )
parser.add_argument('--frame_alpha', default=0.3, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--subword_alpha', default=0.3, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--word_alpha', default=0.3, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--frame_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--subword_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--word_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument(
        "--mask",
        type=str,
        default="",
        help="the way to do frame masking, |random|block|continuous|block_plus|",
    )
parser.add_argument(
        "--multi_level_consistent",
        type=str,
        default="",
        help="the way to do multi level consistent, |similarity|ot|global2local|",
    )
parser.add_argument('--frame_subword_consistent_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--frame_word_consistent_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--subword_word_consistent_weight', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--permutation', action='store_true',
                    help='use permutation')
parser.add_argument('--multi_level_alpha', default=0.3, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--multi_level_ins', default=1, type=float,
                    help='alpha in consistent loss weight',
                    )
parser.add_argument('--permute_probability', action='store_true',
                    help='use permutation probability')
parser.add_argument('--permutation_img_count', default=2, type=int,
                    help='permutation_img_count',
                    )
parser.add_argument('--fw_consist', action='store_true',
                    help='use frame word consistency')
parser.add_argument('--memory_size', default=65536, type=int,
                    help='memory size',
                    )

def main():
    args = parser.parse_args()

    # for OCR 
    if args.model_name[0] == "N":
        args.Transformation = "None"
    elif args.model_name[0] == "T":
        args.Transformation = "TPS"
    elif args.model_name[0] == "S":
        args.Transformation = "SCNN"
    else:
        raise

    if args.model_name[1] == "V":
        args.FeatureExtraction = "ViT"
    elif args.model_name[1] == "R":
        args.FeatureExtraction = "ResNet"
    else:
        raise

    args.SequenceModeling = "BiLSTM"
    args.Prediction = "None"

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

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
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
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
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.model_name))
    model = moco.builder.MoCo(
        Model, args,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    if args.distributed:
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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.useTPS:
        pretrained_state_dict_TPS = torch.load(args.useTPS)
        pretrained_state_dict = {}
        for name in pretrained_state_dict_TPS:
            if "Transformation" in name:
                rename_q = name.replace("module.", "module.encoder_q.")
                rename_k = name.replace("module.", "module.encoder_k.")
                pretrained_state_dict[rename_q] = pretrained_state_dict_TPS[name]
                pretrained_state_dict[rename_k] = pretrained_state_dict_TPS[name]
        for name, param in model.named_parameters():
            try:
                if not name in pretrained_state_dict.keys():
                    raise KeyError
                param.data.copy_(pretrained_state_dict[name].data)  # load from pretrained model                
                param.requires_grad = False  # Freeze
                print(f"TPS layer (freezed): {name}\n")              
            except:
                print(f"non-TPS layer: {name}\n")

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
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
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    if args.light_aug:
        
        if args.data_format == "jpg":
            train_dataset = ImgDataset(
                traindir,
                args
                )
        elif args.data_format == "lmdb":
            train_dataset = LmdbDataset(
                traindir,
                args
            )

    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop((args.imgH, args.imgW), scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        # TODO moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
        if args.data_format == "jpg":
            train_dataset = ImgDataset(
                traindir,
                args,
                transforms=moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
                )
        elif args.data_format == "lmdb":
            train_dataset = ImgDataset(
                traindir,
                args,
                transforms=moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
                )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'model_name': args.model_name,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_name, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    if args.multi_level_consistent == "similarity":            
        losses_frame_subword_consistent = AverageMeter('l_fs', ':.4e')
        losses_frame_word_consistent = AverageMeter('l_fw', ':.4e')
        losses_subword_word_consistent = AverageMeter('l_sw', ':.4e')
    elif args.multi_level_consistent == "global2local": 
        losses_frame_subword_consistent = AverageMeter('l_fs', ':.4e')
        losses_subword_word_consistent = AverageMeter('l_sw', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if args.multi_level_consistent == "similarity": 
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_frame_subword_consistent, losses_frame_word_consistent, losses_subword_word_consistent,top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    elif args.multi_level_consistent == "global2local":
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, losses_frame_subword_consistent, losses_subword_word_consistent,top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        if args.loss_setting == "consistent":
            if args.multi_level_consistent == "similarity":
                frame_output, frame_target, frame_similarity_q, frame_similarity_p, \
                q_similarity_frame_pooled_in_subword, q_similarity_frame_pooled_in_word, \
                subword_output, subword_target, subword_similarity_q, subword_similarity_p, \
                q_similarity_subword, q_similarity_subword_pooled_in_word, \
                word_output, word_target, word_similarity_q, word_similarity_p, \
                q_similarity_word = model(im_q=images[0], im_k=images[1])
                q_similarity_subword_for_frame = q_similarity_subword[:, :q_similarity_frame_pooled_in_subword.size(-1)]
                q_similarity_word_for_frame = q_similarity_word[:, :q_similarity_frame_pooled_in_word.size(-1)]
                q_similarity_word_for_subword = q_similarity_word[:, :q_similarity_subword_pooled_in_word.size(-1)]
                loss_frame_subword_consistent = 0.5 * F.kl_div(q_similarity_frame_pooled_in_subword.log(), q_similarity_subword_for_frame) + \
                    0.5 * F.kl_div(q_similarity_subword_for_frame.log(), q_similarity_frame_pooled_in_subword)
                loss_frame_word_consistent = 0.5 * F.kl_div(q_similarity_frame_pooled_in_word.log(), q_similarity_word_for_frame) + \
                    0.5 * F.kl_div(q_similarity_word_for_frame.log(), q_similarity_frame_pooled_in_word)
                loss_subword_word_consistent = 0.5 * F.kl_div(q_similarity_subword_pooled_in_word.log(), q_similarity_word_for_subword) + \
                    0.5 * F.kl_div(q_similarity_word_for_subword.log(), q_similarity_subword_pooled_in_word)
            elif args.multi_level_consistent == "ot":
                frame_output, frame_target, frame_similarity_q, frame_similarity_p, \
                subword_output, subword_target, subword_similarity_q, subword_similarity_p, \
                word_output, word_target, word_similarity_q, word_similarity_p, \
                loss_frame_subword_consistent = model(im_q=images[0], im_k=images[1])
                loss_frame_word_consistent = 0
                loss_subword_word_consistent = 0
            elif args.multi_level_consistent == "global2local":
                if args.fw_consist:
                    frame_output, frame_target, frame_similarity_q, frame_similarity_p, \
                    subword_output, subword_target, subword_similarity_q, subword_similarity_p, \
                    word_output, word_target, word_similarity_q, word_similarity_p, \
                    fs_output, fs_target, fs_similarity_q, fs_similarity_p, \
                    sw_output, sw_target, sw_similarity_q, sw_similarity_p, \
                    fw_output, fw_target, fw_similarity_q, fw_similarity_p = model(im_q=images[0], im_k=images[1])
                else:
                    frame_output, frame_target, frame_similarity_q, frame_similarity_p, \
                    subword_output, subword_target, subword_similarity_q, subword_similarity_p, \
                    word_output, word_target, word_similarity_q, word_similarity_p, \
                    fs_output, fs_target, fs_similarity_q, fs_similarity_p, \
                    sw_output, sw_target, sw_similarity_q, sw_similarity_p = model(im_q=images[0], im_k=images[1])
                
                loss_fs_ins = criterion(fs_output, fs_target)
                loss_fs_con = 0.5 * F.kl_div(fs_similarity_q.log(), fs_similarity_p) + 0.5 * F.kl_div(fs_similarity_p.log(), fs_similarity_q)
                loss_frame_subword_consistent = args.multi_level_ins * loss_fs_ins + args.multi_level_alpha * loss_fs_con
                loss_sw_ins = criterion(sw_output, sw_target)
                loss_sw_con = 0.5 * F.kl_div(sw_similarity_q.log(), sw_similarity_p) + 0.5 * F.kl_div(sw_similarity_p.log(), sw_similarity_q)
                loss_subword_word_consistent = args.multi_level_ins * loss_sw_ins + args.multi_level_alpha * loss_sw_con
                if args.fw_consist:
                    loss_fw_ins = criterion(fw_output, fw_target)
                    loss_fw_con = 0.5 * F.kl_div(fw_similarity_q.log(), fw_similarity_p) + 0.5 * F.kl_div(fw_similarity_p.log(), fw_similarity_q)
                    loss_frame_word_consistent = args.multi_level_ins * loss_fw_ins + args.multi_level_alpha * loss_fw_con
                else:
                    loss_frame_word_consistent = 0
            elif args.multi_level_consistent == "":
                frame_output, frame_target, frame_similarity_q, frame_similarity_p, \
                subword_output, subword_target, subword_similarity_q, subword_similarity_p, \
                word_output, word_target, word_similarity_q, word_similarity_p = model(im_q=images[0], im_k=images[1])
                loss_frame_subword_consistent = 0
                loss_frame_word_consistent = 0
                loss_subword_word_consistent = 0
            else:
                raise NotImplementedError

            loss_frame_ins = criterion(frame_output, frame_target)
            loss_frame_con = 0.5 * F.kl_div(frame_similarity_q.log(), frame_similarity_p) + 0.5 * F.kl_div(frame_similarity_p.log(), frame_similarity_q)
            loss_subword_ins = criterion(subword_output, subword_target)
            loss_subword_con = 0.5 * F.kl_div(subword_similarity_q.log(), subword_similarity_p) + 0.5 * F.kl_div(subword_similarity_p.log(), subword_similarity_q)
            loss_word_ins = criterion(word_output, word_target)
            loss_word_con = 0.5 * F.kl_div(word_similarity_q.log(), word_similarity_p) + 0.5 * F.kl_div(word_similarity_p.log(), word_similarity_q)
            
            loss = args.frame_weight * loss_frame_ins + args.frame_alpha * loss_frame_con + \
                   args.subword_weight * loss_subword_ins + args.subword_alpha * loss_subword_con + \
                   args.word_weight * loss_word_ins + args.word_alpha * loss_word_con + \
                   args.frame_subword_consistent_weight * loss_frame_subword_consistent + \
                   args.frame_word_consistent_weight * loss_frame_word_consistent + \
                   args.subword_word_consistent_weight * loss_subword_word_consistent
        else:
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(subword_output, subword_target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        if args.multi_level_consistent == "similarity":
            losses_frame_subword_consistent.update(loss_frame_subword_consistent.item(), images[0].size(0))
            losses_frame_word_consistent.update(loss_frame_word_consistent.item(), images[0].size(0))
            losses_subword_word_consistent.update(loss_subword_word_consistent.item(), images[0].size(0))
        elif args.multi_level_consistent == "global2local":
            losses_frame_subword_consistent.update(loss_frame_subword_consistent.item(), images[0].size(0))
            losses_subword_word_consistent.update(loss_subword_word_consistent.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        sys.stdout.flush()


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    print("****************************")
    print("current lr:{}".format(lr))
    print("****************************")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
