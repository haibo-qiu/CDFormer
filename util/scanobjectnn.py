import os
import sys
import h5py
import pickle
import numpy as np
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare_scanobjnn as data_prepare

class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN.
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1],
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882.
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75.h5')

        if not os.path.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)

        if slit_name == 'test' and uniform_sample:
            precomputed_path = os.path.join(
                data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                raise FileExistsError(
                    f'{precomputed_path} does not exist, please compute at first')
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        print(f'Successfully load ScanObjectNN {split} '
              f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        # coord = self.points[idx][:self.num_points]
        coord = self.points[idx]
        label = self.labels[idx]

        if self.partition == 'train':
            np.random.shuffle(coord)

        if self.transform is not None:
            coord, _ = self.transform(coord, coord.copy())  # (coord, feat)

        height = coord[:, self.gravity_dim:self.gravity_dim+1]
        height = height - height.min()

        feat = np.concatenate((coord, height), axis=1)

        # feat = torch.tensor(coord, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        coord = torch.tensor(coord, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return coord, feat, label

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """

if __name__ == '__main__':
    data_dir = 'dataset/scanobjectnn/h5_files/main_split'
    split = 'train'
    num_points = 2048 if split == 'train' else 1024
    dataset = ScanObjectNNHardest(data_dir, split)
    maxs, mins, means = [], [], []
    for i in range(len(dataset)):
        coord, feat, label = dataset.__getitem__(i)
        assert coord.shape[0] == num_points
        print(f'{i:05} shape:{coord.size()}{feat.shape} label:{label}')
        maxs.append(coord.max().item())
        mins.append(coord.min().item())
        means.append(coord.mean().item())
    print(np.max(maxs), np.min(mins), np.mean(means))
