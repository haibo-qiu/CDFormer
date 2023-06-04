#!/bin/bash

set -ex

GPU='0,1,2,3'
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore train_seg.py --config config/s3dis/s3dis_cdformer.yaml debug 0 
