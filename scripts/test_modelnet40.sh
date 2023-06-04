#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls.py --config config/modelnet40/modelnet40_cdformer.yaml \
                                 model_path checkpoints/modelnet40_cdformer.pth
