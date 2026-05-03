#!/bin/bash

# 原版 DETR 的官方推荐参数，为了和您的 YOLO 对比，我们将 epochs 设为 300
# 如果您的显卡(device='1') 显存不够（DETR 非常吃显存），请将 batch_size 从 8 改为 4 或 2

python main.py \
    --dataset_file coco \
    --coco_path /home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD_COCO \
    --output_dir /home/Cug-Rs02/20251011/IRSTD-YOLO/runs/M3FD/train/exp-Original-DETR \
    --epochs 300 \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --device cuda:1