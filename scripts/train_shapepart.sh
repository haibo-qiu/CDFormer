#!/bin/bash

set -ex

GPU='0,1,2,3'
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore train_partseg.py --config config/shapenetpart/shapenetpart_cdformer.yaml debug 0
