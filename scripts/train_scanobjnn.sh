#!/bin/bash

set -ex

GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore main_cls.py --config config/scanobjectnn/scanobjectnn_cdformer.yaml debug 0
