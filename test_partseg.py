import os
import time
import datetime
import random
import numpy as np
import argparse
import importlib

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import dataset, config
from util.shapenetpart import ShapeNetPartNormal
from util.common_util import set_random_seed, create_eval_log, create_test_log
from util.common_util import part_seg_refinement, get_ins_mious
from util.data_util import collate_fn_shapenetpart
from util import transform

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


def main():
    args = get_parser()
    if args.manual_seed is not None:
        set_random_seed(args.manual_seed)

    if 'checkpoints' in args.model_path:
        logger = create_eval_log(args.model_path)
    else:
        flag = 'best' if 'best' in args.model_path else 'last'
        logger = create_test_log(args.save_path, flag=flag)

    Net = importlib.import_module(f'model.{args.net}')
    cls_dim = args.get('cls_dim', 64)
    mid_dim = args.get('mid_dim', 256)
    final_dim = args.get('final_dim', 64)
    model = Net.CDFormer(downscale=args.downsample_scale, num_heads=args.num_heads, depths=args.depths, channels=args.channels, k=args.k,
                            up_k=args.up_k, drop_path_rate=args.drop_path_rate, ratio=args.ratio, num_layers=args.num_layers, cls_dim=cls_dim, final_dim=final_dim,
                            mid_dim=mid_dim, use_cls=args.use_cls, concat_xyz=args.concat_xyz, fea_dim=args.fea_dim, num_classes=args.classes, stem_transformer=args.stem_transformer)
    logger.info(args)
    logger.info(model)
    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    model = torch.nn.DataParallel(model.cuda())

    val_data = ShapeNetPartNormal(data_root=args.data_root, num_points=args.num_points, split='test', presample=True, transform=None)
    args.cls2parts = val_data.cls2parts

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size_val,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=None,
                                             collate_fn=collate_fn_shapenetpart)
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{} for testing'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        best_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f'Load best ins iou {best_iou} in epoch {best_epoch}')
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.model_path))
        return

    jitter_sigma = args.get('jitter_sigma', 0.001)
    jitter_clip = args.get('jitter_clip', 0.005)
    vote_transform = transform.Compose([
        transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
        transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
        transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
    ])
    test_ins_miou, test_cls_miou, test_cls_mious = validate(val_loader, model, logger, args, args.num_votes, vote_transform=vote_transform)


def validate(val_loader, model, logger, args, num_votes=1, vote_transform=None):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    cls_mious = torch.zeros(args.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(args.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    torch.cuda.empty_cache()

    coord_list = []
    pred_list = []
    target_list = []
    model.eval()
    pbar = tqdm(total=len(val_loader))
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
        coord_list.append(coord)
        pred_list.append(preds)
        target_list.append(target)
        batch_ins_mious = get_ins_mious(preds, target, cls, args.cls2parts)
        ins_miou_list += batch_ins_mious

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1
        pbar.update(1)

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()

    for cat_idx in range(args.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    if num_votes > 0:
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
