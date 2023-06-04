#!/bin/bash
set -ex

# eval pretrained model
GPU=$1
CUDA_VISIBLE_DEVICES=$GPU \
python -u -W ignore test_seg.py --config config/s3dis/s3dis_cdformer.yaml \
                                  model_path checkpoints/s3dis_cdformer.pth \
                                  save_folder 'temp/s3dis/results'


# eval after training
#Results='output/s3dis/cdformer/2023-10-10-10-10-10'
#GPU='0'
#CUDA_VISIBLE_DEVICES=$GPU \
#python -u -W ignore test_seg.py --config $Results/cfg.yaml \
                                  #save_path $Results \
                                  #model_path $Results/model/model_best.pth \
                                  #save_folder $Results/results/best \
