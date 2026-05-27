#!/bin/bash
# Experiment 6: Best Exp5 config + EMA (decay=0.999)
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
  --ema-decay 0.999 \
  --output-dir outputs/exp6_wider_ema