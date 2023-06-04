import os
import time
import datetime
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
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, config
from util.shapenetpart import ShapeNetPartNormal
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate, smooth_loss
from util.common_util import code_backup, create_log, get_git_revision_hash, set_random_seed
from util.common_util import create_save_path, get_log, part_seg_refinement, get_ins_mious
from util.data_util import collate_fn_shapenetpart, batch_to_stack, stack_to_batch
from util.adan import Adan
from util import transform
from util.logger import get_logger

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, CosineAnnealingWarmupRestarts
from util.cross_entropy import SmoothCrossEntropy, Poly1FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch_points_kernels as tp
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description='CDFormer For Point Cloud Part Segmentation')
    parser.add_argument('--config', type=str, default='config/shapenetpart/shapenetpart_cdformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/shapenetpart/shapenetpart_cdformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
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
    if args.manual_seed is not None:
        set_random_seed(args.manual_seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.dist_url == "env://":
        torch.distributed.barrier()
    args.cur_time = datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M")
    args.save_path = create_save_path(args.save_path, args.cur_time, args.debug)

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
    global args, best_iou
    args, best_iou = argss, 0

    if args.distributed:
        if args.dist_url == "env://":
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)

    if main_process():
        global logger, writer
        # logger, args.save_path = create_log(args.save_path, args.debug)
        logger = get_log(args.save_path)
        with open(os.path.join(args.save_path, 'cfg.yaml'), 'w') as fp:
            yaml.dump(dict(args), fp)
        code_backup(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> current git commit: {}".format(get_git_revision_hash()))
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

        # init wandb
        if args.wandb and args.debug == 0:
            wandb.init(
                    project="cdformer_shapenetpart",
                    name=os.path.basename(args.save_path),
                    config=dict(args),
                    sync_tensorboard=True
                    )

    # get model
    # from model.tat_posi import Stratified
    Net = importlib.import_module(f'model.{args.net}')
    cls_dim = args.get('cls_dim', 48)
    mid_dim = args.get('mid_dim', 256)
    final_dim = args.get('final_dim', 64)
    model = Net.CDFormer(downscale=args.downsample_scale, num_heads=args.num_heads, depths=args.depths, channels=args.channels, k=args.k,
                           up_k=args.up_k, drop_path_rate=args.drop_path_rate, ratio=args.ratio, num_layers=args.num_layers, cls_dim=cls_dim, final_dim=final_dim,
                           mid_dim=mid_dim, use_cls=args.use_cls, concat_xyz=args.concat_xyz, fea_dim=args.fea_dim, num_classes=args.classes, stem_transformer=args.stem_transformer)

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model.cuda())

    val_transform = None
    if args.data_name == 'shapenetpart':
        val_data = ShapeNetPartNormal(data_root=args.data_root, num_points=args.num_points, split='test', presample=True, transform=val_transform)
        args.cls2parts = val_data.cls2parts
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_shapenetpart)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'shapenetpart':
        train_transform = None
        if args.aug:
            jitter_sigma = args.get('jitter_sigma', 0.01)
            jitter_clip = args.get('jitter_clip', 0.05)
            if main_process():
                logger.info("augmentation all")
                logger.info("jitter_sigma: {}, jitter_clip: {}".format(jitter_sigma, jitter_clip))
            train_transform = transform.Compose([
                transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
                transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
                transform.RandomDropColor(color_augment=args.get('color_augment', 0.0)),
            ])
        train_data = ShapeNetPartNormal(data_root=args.data_root, num_points=args.num_points, split='trainval', transform=train_transform)
        args.cls2parts = train_data.cls2parts
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.loss == 'SmoothCrossEntropy':
        criterion = SmoothCrossEntropy(label_smoothing=args.label_smoothing, ignore_index=args.ignore_label, num_classes=args.classes).cuda()
    elif args.loss == 'Poly1FocalLoss':
        criterion = Poly1FocalLoss().cuda()
    else:
        raise ValueError("The loss {} is not supported.".format(args.loss))

    if main_process():
        logger.info("criterion: {}".format(criterion))
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn_shapenetpart)


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
        milestones = args.milestones if hasattr(args, "milestones") else [int(args.epochs*0.6), int(args.epochs*0.8)]
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
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*iter_per_epoch, eta_min=args.base_lr / 10000.)
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
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            val_ins_miou, val_cls_miou, val_cls_mious = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('cls_miou_val', val_cls_miou, epoch_log)
                writer.add_scalar('ins_miou_val', val_ins_miou, epoch_log)
                is_best = val_ins_miou > best_iou
                best_iou = max(best_iou, val_ins_miou)
                if args.wandb and args.debug == 0:
                    wandb.run.summary["best_iou"] = best_iou
                logger.info('Current best ins miou: {:.5f}, is_best: {}'.format(best_iou, is_best))
                writer.add_scalar('best_ins_miou', best_iou, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
        # break
    torch.distributed.barrier()
    if args.get('num_votes', 0) > 0:
        model_path = os.path.join(args.save_path, 'model/model_best.pth')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
        best_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        if main_process():
            logger.info(f'Load best ins iou {best_iou} in epoch {best_epoch}')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        set_random_seed(args.manual_seed)
        vote_transform = transform.Compose([
            transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
            transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
            transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
        ])
        test_ins_miou, test_cls_miou, test_cls_mious = validate(val_loader, model, criterion, args.num_votes, vote_transform=vote_transform)

    if main_process():
        writer.close()
        cost = (time.time() - start_time) / 3600.
        logger.info('==>Total training time: {:.2f} hours'.format(cost))


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, feat, target, cls, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        cls = cls.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(feat, coord, offset, batch, neighbor_idx, cls)
            assert output.shape[1] == args.classes
            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls
            loss = criterion(output, target)

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
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

    return loss_meter.avg


def validate(val_loader, model, criterion, num_votes=1, vote_transform=None):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    cls_mious = torch.zeros(args.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(args.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, cls, offset) in enumerate(val_loader):
        batch_size = len(offset)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]

        target, offset = target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        coord, feat = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True),
        cls = cls.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        logits = 0
        for v in range(num_votes):
            set_random_seed(v)
            if v > 0:
                coord, _ = vote_transform(coord.cpu().numpy(), None)
                coord = torch.from_numpy(coord).cuda(non_blocking=True)
            if args.concat_xyz:
                feats = torch.cat([feat, coord], 1)

            with torch.no_grad():
                logits += model(feats, coord, offset, batch, neighbor_idx, cls)
        logits /= num_votes
        preds = logits.max(dim=1)[1]
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        coord = coord.reshape(batch_size, -1, coord.shape[-1])

        if args.get('refine', False):
            part_seg_refinement(preds, coord, cls, args.cls2parts, args.get('refine_n', 10))
        batch_ins_mious = get_ins_mious(preds, target, cls, args.cls2parts)
        ins_miou_list += batch_ins_mious

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info(f'{i+1} / {len(val_loader)}')

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if args.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(args.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    if main_process():
        if num_votes > 1:
            logger.info('=============Voting==============')
        logger.info(f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return ins_miou, cls_miou, cls_mious

if __name__ == '__main__':
    import gc
    gc.collect()
    main()
