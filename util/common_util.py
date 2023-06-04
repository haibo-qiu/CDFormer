import os
import random
import shutil
import logging
import subprocess
import numpy as np
import datetime
from PIL import Image
from collections import Counter

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


@torch.no_grad()
def knn_point(k, query, support=None):
    """Get the distances and indices to a fixed number of neighbors
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
    """
    if support is None:
        support = query
    dist = torch.cdist(query, support)
    k_dist = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return k_dist.values, k_dist.indices


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


def set_random_seed(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def find_unused_params(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def code_backup(log_path):

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    pwd = os.getcwd()
    ignores = ['.git', '.gitignore', 'debug', 'dataset', 'logs', 'pretrained', '__pycache__', 'output', 'lib']
    valid_ext = ['.py', '.sh', '.ply', '.yaml', '.pbs']
    try:
        print("Copying files to %s for further reference." % log_path)
        log_path = os.path.join(log_path, "code_bup")
        for v in sorted(os.listdir(pwd)):
            if v not in ignores:
                if not os.path.isdir(os.path.join(pwd, v)):
                    if any(v.endswith(ext) for ext in valid_ext) and 'debug' not in v:
                        os.makedirs(log_path, exist_ok=True)
                        shutil.copyfile(os.path.join(pwd, v), os.path.join(log_path, v))
                else:
                    for dp, dn, fn in os.walk(os.path.join(pwd, v)):
                        for f in fn:
                            if any(f.endswith(ext) for ext in valid_ext):
                                filename = os.path.join(dp, f)
                                os.makedirs(os.path.dirname(filename.replace(pwd, log_path)), exist_ok=True)
                                shutil.copyfile(filename, filename.replace(pwd, log_path))
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

def mp_logger(meg, name='main-logger'):
    if dist.get_rank() == 0:
        logger = logging.getLogger(name)
        logger.info(meg)

def create_test_log(log_path, flag='best', name='main-logger'):
    # create logger
    cur_time = datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S")
    log_file = os.path.join(log_path, f'testing_{flag}_{cur_time}.log')
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    return logger

def create_eval_log(checkpoint='shapenetpart_cdformer.pth', name='main-logger'):
    log_path = os.path.basename(checkpoint).split('.')[0]
    dataset, model = log_path.split('_')
    log_path = os.path.join('output', dataset, model, 'eval')

    cur_time = datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S")

    # create log folder
    try:
        os.makedirs(log_path, exist_ok=True)
    except Exception as e:
        print(e)
        quit()

    log_file = os.path.join(log_path, f'eval_{cur_time}.log')
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    return logger


def create_save_path(log_path='output/s3dis/tat/', cur_time=None, debug=False, eval=False):
    if eval:
        log_path = os.path.join(log_path, 'eval')
    elif debug:
        log_path = os.path.join(log_path, 'debug')

    # cur_time = datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S")
    assert cur_time
    log_path = os.path.join(log_path, cur_time)

    # create log folder
    try:
        # if os.path.isdir(log_path):
            # shutil.rmtree(log_path)
        os.makedirs(os.path.join(log_path, 'model'), exist_ok=True)
        os.makedirs(os.path.join(log_path, 'results', 'best'), exist_ok=True)
        os.makedirs(os.path.join(log_path, 'results', 'last'), exist_ok=True)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()
    return log_path

def get_log(log_path, name='main-logger'):
    # create logger
    log_file = os.path.join(log_path, 'training.log')
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    return logger


def create_log(log_path='output/s3dis/stratified_transformer/', debug=False, eval=False, name='main-logger'):
    if eval:
        log_path = os.path.join(log_path, 'eval')
    elif debug:
        log_path = os.path.join(log_path, 'debug')

    cur_time = datetime.datetime.now().strftime("%Y-%-m-%d-%H-%M-%S")
    log_path = os.path.join(log_path, cur_time)

    # create log folder
    try:
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        os.makedirs(os.path.join(log_path, 'model'), exist_ok=True)
        os.makedirs(os.path.join(log_path, 'results', 'best'), exist_ok=True)
        os.makedirs(os.path.join(log_path, 'results', 'last'), exist_ok=True)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # create logger
    log_file = os.path.join(log_path, 'training.log')
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # add file handler to save the log to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False

    return logger, log_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split('.')) > 1:
            recursive_set(getattr(cur_module, name[:name.find('.')]), name[name.find('.')+1:], module)
        else:
            setattr(cur_module, name, module)
    from lib.sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(model, name, SynchronizedBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(model, name, SynchronizedBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(model, name, SynchronizedBatchNorm3d(m.num_features, m.eps, m.momentum, m.affine))


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def memory_use():
    BYTES_IN_GB = 1024 ** 3
    return 'ALLOCATED: {:>6.3f} ({:>6.3f})  CACHED: {:>6.3f} ({:>6.3f})'.format(
        torch.cuda.memory_allocated() / BYTES_IN_GB,
        torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        torch.cuda.memory_reserved() / BYTES_IN_GB,
        torch.cuda.max_memory_reserved() / BYTES_IN_GB,
    )


def smooth_loss(output, target, eps=0.1):
    w = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = F.log_softmax(output, dim=1)
    loss = (-w * log_prob).sum(dim=1).mean()
    return loss
