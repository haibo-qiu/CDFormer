import os
import sys
import numpy as np
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import collate_fn
from util.data_util import data_prepare_s3dis as data_prepare

s3dis_names=['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
             'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        self.data_root = data_root
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + '.npy')
        data = np.load(data_path)

        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == '__main__':
    data_root = 'dataset/s3dis/stanford_indoor3d/'
    test_area, voxel_size, voxel_max = 5, 0.04, 80000

    point_data = S3DIS(split='train', data_root=data_root, test_area=test_area, voxel_size=voxel_size, voxel_max=voxel_max)
    print('point data size:', point_data.__len__())
    import torch, time, random
    from util.vis_util import write_ply_color, write_ply_rgb
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            write_ply_color(coord.numpy(), label.numpy(), out_filename='temp/vis_test_color.obj', num_classes=13)
            write_ply_rgb(coord.numpy(), 255*feat.numpy(), out_filename='temp/vis_test_real.obj', num_classes=13)
            voxel_num.append(label.shape[0])
            end = time.time()
            assert False
    print(np.sort(np.array(voxel_num)))
