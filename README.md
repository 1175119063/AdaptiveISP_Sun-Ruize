# AdaptiveISP 实验复现与改动 

## 环境要求

- Ubuntu 22.04, NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- Miniconda, Python 3.10, PyTorch 2.0.1+cu118
- 依赖安装：在 AdaptiveISP 根目录运行 `bash setup.sh`

## 目录结构

```
adaptive ISP/
├── AdaptiveISP/              ← 开源代码主目录
├── AdaptiveISP_Sun Ruize/    ← 本次提交（本目录）
│   ├── README.md             ← 本文件
│   ├── report.pdf            ← 实验报告
│   ├── config.py             ← 修改后的配置文件
│   ├── bird.yaml             ← BIRDS 数据集配置
│   ├── run_1_train.sh        ← 训练脚本
│   ├── run_2_eval_all.sh     ← 全部评测脚本
│   └── results/              ← 实验结果摘要
├── datasets/BIRDS/PNG/       ← 自建 BIRDS 数据集
└── pretrained/               ← 预训练权重
```

## 数据集

BIRDS 数据集共 **500 张鸟类照片，全部来自个人拍摄**（Sony α 系列相机，ARW 格式）。拍摄场景涵盖白天自然光、林间阴影、水面反光、逆光等多种光照条件。

### 数据集下载

已将处理后的 PNG 数据集上传至百度网盘：

> **链接**: https://pan.baidu.com/s/1S4QrRCEjuppy4dvwpwlutQ?pwd=4m3j
> **提取码**: 4m3j

下载后解压到 `datasets/BIRDS/PNG/`，目录结构如下：

```
datasets/BIRDS/PNG/
├── images/
│   ├── train/  (400 张)
│   └── val/    (100 张)
├── labels/
│   ├── train/  (400 个 .txt, YOLOv8 pseudo-label)
│   └── val/    (100 个 .txt)
├── train.txt   (图片路径列表)
└── val.txt     (图片路径列表)
```

### 数据集制作流程

1. 500 张 ARW 原始照片 → `rawpy` 读取 + gamma 校正 → 导出为 PNG
2. 使用 YOLOv8x 预训练模型对 PNG 进行推理，导出 YOLO 格式 pseudo-label
3. 按 400/100 划分训练集和验证集

## 预训练权重下载

```bash
mkdir -p ../pretrained
cd ../pretrained

# YOLOv3 检测器
wget https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/yolov3.pt

# LOD-Agent ISP 模型
wget https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/ckpt-lod-df-1.0.pth
wget https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/ckpt-lod-df-0.98.pth
```

## 运行命令

以下脚本均在 `AdaptiveISP/` 目录下运行：

### 1. 训练 Bird-Agent（可选，已有预训练权重可跳过）

```bash
cd AdaptiveISP
bash ../AdaptiveISP_Sun\ Ruize/run_1_train.sh
```

约 1 小时 40 分钟，显存占用 ~3.5GB。

### 2. 评测（全部三组对比）

```bash
cd AdaptiveISP
bash ../AdaptiveISP_Sun\ Ruize/run_2_eval_all.sh
```

三组评测：
- LOD-Agent (steps=5) → BIRDS val 100
- Bird-Agent (steps=5) → BIRDS val 100
- Bird-Agent (steps=3) → BIRDS val 100

结果分别保存在 `my_test/my_test_on_BIRDS_5step/{lod_agent_eval,bird_agent_eval}/` 和 `my_test/my_test_on_BIRDS_3step/bird_agent_3step/`。

## 实验结果速览

| 模型 | mAP50 | mAP75 | mAP50-95 | Precision | Recall | 推理时间 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LOD-Agent (steps=5) | 0.738 | 0.579 | **0.551** | 0.735 | 0.698 | 374.4ms |
| Bird-Agent (steps=5) | 0.738 | 0.566 | 0.529 | 0.762 | 0.683 | 374.9ms |
| Bird-Agent (steps=3) | 0.734 | 0.553 | 0.521 | 0.749 | 0.679 | **227.6ms** |

## 主要发现

1. **微调未提升性能**：在 BIRDS 上微调的 Bird-Agent (mAP50-95=0.529) 不如 LOD-Agent (0.551)。分析滤波器分布发现，Bird-Agent 过拟合到 Gamma+SaturationPlus 捷径（占 40%），导致 Recall 下降。

2. **步数可压缩**：steps=5→3，mAP50-95 仅降 0.008，推理加速 39%。3 步 ISP 管线已接近最优。

3. **数据多样性 > Domain Match**：LOD 多场景数据训练的鲁棒策略优于 BIRDS 单场景微调。

详见 `report.pdf`。
