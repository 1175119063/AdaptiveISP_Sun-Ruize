#!/bin/bash
# ============================================================
# 实验 2：三组评测对比
# 1. LOD-Agent (预训练) + steps=5
# 2. Bird-Agent (微调)  + steps=5
# 3. Bird-Agent (微调)  + steps=3
# 结果分别保存在 my_test/ 下对应目录
# ============================================================

cd "$(dirname "$0")/.."
cd AdaptiveISP

eval "$(conda shell.bash hook)"
conda activate adaptiveisp

echo "=============================================="
echo "评测 1/3: LOD-Agent (steps=5) → BIRDS val 100"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python yolov3/val_adaptiveisp.py \
    --data=yolov3/data/bird.yaml \
    --data_name=lod \
    --isp_model=Agent \
    --isp_weights=../pretrained/ckpt-lod-df-1.0.pth \
    --weights=../pretrained/yolov3.pt \
    --cfg_file=config \
    --project="../my_test/my_test_on_BIRDS" \
    --name=lod_agent_eval \
    --batch-size=1 \
    --steps=5 \
    --save_image \
    --save_param \
    --exist-ok

echo ""
echo "=============================================="
echo "评测 2/3: Bird-Agent (steps=5) → BIRDS val 100"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python yolov3/val_adaptiveisp.py \
    --data=yolov3/data/bird.yaml \
    --data_name=lod \
    --isp_model=Agent \
    --isp_weights=experiments/lod-bird_agent/ckpt/DynamicISP_iter_10000.pth \
    --weights=../pretrained/yolov3.pt \
    --cfg_file=config \
    --project="../my_test/my_test_on_BIRDS" \
    --name=bird_agent_eval \
    --batch-size=1 \
    --steps=5 \
    --save_image \
    --save_param \
    --exist-ok

echo ""
echo "=============================================="
echo "评测 3/3: Bird-Agent (steps=3) → BIRDS val 100"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 python yolov3/val_adaptiveisp.py \
    --data=yolov3/data/bird.yaml \
    --data_name=lod \
    --isp_model=Agent \
    --isp_weights=experiments/lod-bird_agent/ckpt/DynamicISP_iter_10000.pth \
    --weights=../pretrained/yolov3.pt \
    --cfg_file=config \
    --project="../my_test/my_test_on_BIRDS_3step" \
    --name=bird_agent_3step \
    --batch-size=1 \
    --steps=3 \
    --save_image \
    --save_param \
    --exist-ok

echo ""
echo "=============================================="
echo "全部评测完成！"
echo "结果目录："
echo "  my_test/my_test_on_BIRDS/lod_agent_eval/"
echo "  my_test/my_test_on_BIRDS/bird_agent_eval/"
echo "  my_test/my_test_on_BIRDS_3step/bird_agent_3step/"
echo "=============================================="
