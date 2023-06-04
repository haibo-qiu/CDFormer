import os
import time
import yaml
import random
import numpy as np
import logging
import argparse
import shutil
import wandb
import importlib

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torch_points_kernels as tp
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from util import dataset, config
from util.scanobjectnn import ScanObjectNNHardest
from util.modelnet40 import ModelNet40Ply2048
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate
from util.common_util import code_backup, create_log, create_test_log, create_eval_log, get_git_revision_hash
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, CosineAnnealingWarmupRestarts
from util.cross_entropy import SmoothCrossEntropy
from util.data_util import batch_to_stack, cutmix
from util.adan import Adan
from util import transform
from util.logger import get_logger

from tqdm import tqdm
from tensorboardX import SummaryWriter
from functools import partial


def get_parser():
    parser = argparse.ArgumentParser(description='CDFormer For Point Cloud Classification')
    parser.add_argument('--config', type=str, default='config/modelnet40/modelnet40_cdformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/modelnet40/modelnet40_cdformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return dist.get_rank() == 0


def main():
    args = get_parser()
    if args.manual_seed is None:
        args.manual_seed = np.random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multi_node:
        args.dist_url = "env://"

    if args.dist_url == "env://":
        rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ["WORLD_SIZE"])
        main_worker(rank, args)
    elif args.dist_url.startswith('tcp://localhost'):
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        raise NotImplementedError()


def main_worker(rank, argss):
    global args, best_acc
    args, best_acc = argss, 0

    if args.distributed:
        if args.dist_url == "env://":
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)

    if main_process():
        global logger, writer
        if args.model_path:
            if 'checkpoints' in args.model_path:
                logger = create_eval_log(args.model_path)
            else:
                flag = 'best' if 'best' in args.model_path else 'last'
                logger = create_test_log(args.save_path, flag=flag)
        else:
            logger, args.save_path = create_log(args.save_path, args.debug)
            with open(os.path.join(args.save_path, 'cfg.yaml'), 'w') as fp:
                yaml.dump(dict(args), fp)
            # code_backup(args.save_path)
            writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> current git commit: {}".format(get_git_revision_hash()))
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

        # init wandb
        if args.wandb and args.debug == 0:
            wandb.init(project='_'.join([args.arch, args.data_name]),
                       name=os.path.basename(args.save_path),
                       config=dict(args),
                       sync_tensorboard=True)

    # get model
    Net = importlib.import_module(f'model.{args.net}')
    model = Net.CDFormer(downscale=args.downsample_scale, num_heads=args.num_heads, depths=args.depths, channels=args.channels, k=args.k,
                            up_k=args.up_k, drop_path_rate=args.drop_path_rate, ratio=args.ratio, num_layers=args.num_layers,
                            concat_xyz=args.concat_xyz, num_classes=args.classes, stem_transformer=args.stem_transformer, fea_dim=args.fea_dim)

    if main_process():
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adan':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = Adan(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)


    if args.distributed:
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        find_unused = False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=find_unused)
    else:
        model = torch.nn.DataParallel(model.cuda())


    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint['best_acc']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'scanobjectnn':
        train_transform = None
        if args.aug:
            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            scale_ratio = args.get('scale_ratio', 0.2)
            shift_range = args.get('shift_range', 0.2)
            train_transform = transform.Compose([
                transform.RandomDiffScale(scale_low=1/(1+scale_ratio), scale_high=1+scale_ratio),
                transform.RandomShift(shift_range=shift_range),
                transform.RandomRotate(along_z=args.get('rotate_along_z', False)),
            ])
        train_data = ScanObjectNNHardest(data_dir=args.data_root, split=args.train_split, transform=train_transform)
    elif args.data_name == 'modelnet40':
        train_transform = None
        if args.aug:
            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            scale_ratio = args.get('scale_ratio', 0.2)
            shift_range = args.get('shift_range', 0.2)
            train_transform = transform.Compose([
                transform.RandomDiffScale(scale_low=1/(1+scale_ratio), scale_high=1+scale_ratio),
                transform.RandomShift(shift_range=shift_range),
            ])
        train_data = ModelNet40Ply2048(data_dir=args.data_root, split=args.train_split, transform=train_transform)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    criterion = SmoothCrossEntropy(ignore_index=args.ignore_label, num_classes=args.classes, label_smoothing=args.label_smoothing).cuda()

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
        logger.info("criterion: {}".format(criterion))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True)

    val_transform = None
    if args.data_name == 'scanobjectnn':
        val_data = ScanObjectNNHardest(data_dir=args.data_root, split=args.test_split, transform=val_transform)
    elif args.data_name == 'modelnet40':
        val_data = ModelNet40Ply2048(data_dir=args.data_root, split=args.test_split, transform=val_transform)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']
            if main_process():
                logger.info("=> loaded weight '{}' with acc {}".format(args.weight, best_acc))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.model_path:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']
            if main_process():
                logger.info("=> loaded trained model '{}' with acc {}".format(args.model_path, best_acc))
            validate(val_loader, model, criterion)
            return
        else:
            logger.info("=> no trained model found at '{}'".format(args.model_path))
            return

    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        milestones = [int(args.epochs*0.6), int(args.epochs*0.8)]
        gamma = args.gamma if hasattr(args, 'gamma') else 0.1
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    elif args.scheduler == 'Cosine':
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: CosineAnnealing. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*iter_per_epoch, eta_min=args.base_lr / 1000.)
    elif args.scheduler == 'CosineWarmup':
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: CosineAnnealingWarmUp. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        cycle_steps = int(args.warmup_cycle_ratio*args.epochs*iter_per_epoch)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cycle_steps)
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    start_time = time.time()
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = allAcc_val > best_acc
                if is_best:
                    best_acc = allAcc_val
                    best_epoch = epoch_log
                if args.wandb and args.debug == 0:
                    wandb.run.summary["best_acc"] = best_acc
                logger.info('Current best all acc: {:.5f} at epoch {}'.format(best_acc, best_epoch))

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_acc': best_acc, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        acc_log = os.path.join(args.save_path, 'training_{:.4f}.log'.format(best_acc))
        shutil.move(os.path.join(args.save_path, 'training.log'), acc_log)
        cost = (time.time() - start_time) / 3600.
        logger.info('==>Total training time: {:.2f} hours'.format(cost))
        logger.info('==>Training done!\nBest Overall Acc: %.3f' % (best_acc))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)

        coord, feat, target = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)

        if args.num_sample_points < coord.shape[1]:
            fps_idx = tp.furthest_point_sample(coord, args.num_sample_points)
            fps_idx = fps_idx[:, np.random.choice(args.num_sample_points, args.num_points, False)]

            coord = torch.gather(coord, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, coord.shape[-1]))
            feat = torch.gather(feat, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, feat.shape[-1]))

        coord, feat, target = cutmix(coord, feat, target, cutmix_prob=args.cutmix_prob, num_points=args.num_points)
        coord, feat, offset = batch_to_stack(coord, feat)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
        batch = batch.cuda(non_blocking=True)

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

        offset = offset.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(feat, coord, offset, batch, neighbor_idx)
            assert output.shape[1] == args.classes
            if type(target) == list:
                target_a, target_b, lam = target
                target_a, target_b = target_a.cuda(non_blocking=True), target_b.cuda(non_blocking=True)
                loss = criterion(output, target_a) * (1-lam) + criterion(output, target_b) * lam
                target = target_a
            else:
                target = target.cuda(non_blocking=True)
                loss = criterion(output, target)

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                          batch_time=batch_time, data_time=data_time,
                                          remain_time=remain_time,
                                          loss_meter=loss_meter,
                                          lr=lr))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        coord, feat, offset = batch_to_stack(coord, feat)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        with torch.no_grad():
            output = model(feat, coord, offset, batch, neighbor_idx)
            loss = criterion(output, target)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: allAcc/mAcc {:.3f}/{:.3f}.'.format(allAcc, mAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: accuracy {:.4f}.'.format(i, accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
