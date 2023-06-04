#!/bin/bash
set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore test_partseg.py --config config/shapenetpart/shapenetpart_cdformer.yaml \
                                         batch_size_val 80 \
                                         model_path checkpoints/shapenetpart_cdformer.pth
