#!/bin/bash
# ============================================================
# 实验 1：在 BIRDS 数据集上微调 Bird-Agent
# 从 LOD 预训练权重 ckpt-lod-df-1.0.pth 初始化
# 显存占用约 3.5GB，训练时长约 1 小时 40 分钟
# ============================================================

cd "$(dirname "$0")/.."
cd AdaptiveISP

eval "$(conda shell.bash hook)"
conda activate adaptiveisp

CUDA_VISIBLE_DEVICES=0 python train.py \
    --batch_size=2 \
    --data_name=lod \
    --data_cfg=yolov3/data/bird.yaml \
    --save_path=lod-bird_agent \
    --max_iters=10000 \
    --lr=1.5e-5

echo "训练完成！Checkpoint 保存在 experiments/lod-bird_agent/ckpt/"
