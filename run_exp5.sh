#!/bin/bash
# Experiment 5: Wider ResNet (width=48) + ColorJitter augmentation + 100 epochs
# Expected improvement: larger model capacity + mild augmentation
cd ~/repo_v2
export PATH=/home/yikai/miniconda3/bin:$PATH && source /home/yikai/miniconda3/etc/profile.d/conda.sh && conda activate base
python train_siglip.py \
  --data-dir Flickr8k \
  --text-encoder transformer \
  --embed-dim 256 \
  --image-width 48 \
  --num-heads 4 \
  --num-layers 2 \
  --epochs 100 \
  --batch-size 256 \
  --lr 3e-4 \
  --warmup-ratio 0.1 \
  --min-lr-ratio 0.01 \
  --output-dir outputs/exp5_wider_resnet_colorjitter