import random
import torch
import numpy as np

from util.voxelize import voxelize


def cutmix(points, feat, target, beta=1, cutmix_prob=0.5, num_points=1024):
    import lib.emd.emd_module as emd
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        lam = np.random.beta(beta, beta)
        B = points.size()[0]

        rand_index = torch.randperm(B).cuda()
        target_a = target
        target_b = target[rand_index]

        point_a = points
        point_b = points[rand_index]
        point_c = points[rand_index]
        feat_c = feat[rand_index]

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(point_a, point_b, 0.005, 300)
        for ass in range(B):
            point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]
            feat_c[ass, :, :] = feat_c[ass, ind[ass].long(), :]

        int_lam = int(num_points * lam)
        int_lam = max(1, int_lam)

        random_point = torch.from_numpy(np.random.choice(num_points, B, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(B))
        query = point_a[ind1, random_point].view(B, 1, 3)
        dist = torch.sqrt(torch.sum((point_a - query.repeat(1, num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(B):
            points[i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
            feat[i2, idxs[i2], :] = feat_c[i2, idxs[i2], :]
        # adjust lambda to exactly match point ratio
        lam = int_lam * 1.0 / num_points
        return points, feat, [target_a, target_b, lam]
    return points, feat, target


def collate_fn_limit(batch, max_batch_points, logger):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.tensor(offset[:k], dtype=torch.int32)


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def collate_fn_cls(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.tensor(label, dtype=torch.long), torch.tensor(offset, dtype=torch.int)


def collate_fn_shapenetpart(batch):
    coord, feat, label, cls = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.cat(cls), torch.IntTensor(offset)


def stack_to_batch(batch_size, v_list):
    results = []
    for v in v_list:
        print(v.shape)
        c = v.shape[-1]
        v = v.reshape(batch_size, -1, c)
        results.append(v)
    return results

def batch_to_stack(coord, feat):
    # coord: (B, N, 3)
    # feat: (B, N, 4)
    B, N, _ = coord.shape
    coord = coord.reshape(B*N, -1)
    feat = feat.reshape(B*N, -1)
    offset = [N * (i+1) for i in range(B)]
    return coord, feat, torch.IntTensor(offset)

def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def data_prepare_modelnet40(coord, feat, label, split='train', voxel_size=0.04, voxel_min=1024, transform=None, shuffle_index=False):
    if transform:
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        while len(uniq_idx) < voxel_min:
            voxel_size /= 2
            uniq_idx = voxelize(coord, voxel_size)
        coord, feat = coord[uniq_idx], feat[uniq_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.tensor(coord, dtype=torch.float32)
    feat = torch.tensor(feat, dtype=torch.float32) / 255
    label = torch.tensor(label, dtype=torch.long)
    return coord, feat, label


def data_prepare_s3dis(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.tensor(coord, dtype=torch.float32)
    feat = torch.tensor(feat, dtype=torch.float32) / 255
    label = torch.tensor(label, dtype=torch.long)
    return coord, feat, label


def data_prepare_scanobjnn(coord, feat, split='train', transform=None, shuffle_index=False):
    if transform:
        coord, feat = transform(coord, feat)
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    return coord, feat

