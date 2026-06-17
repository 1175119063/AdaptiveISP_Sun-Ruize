#!/usr/bin/env python3
"""
自定义图片 ISP 推理 + YOLOv3 检测
输出:
    img_results/step-0/ ~ step-5/  每步处理结果
    param_results/*.json           ISP参数
    records.txt                   滤波器序列
    detect_before/                处理前YOLO检测图
    detect_after/                 处理后YOLO检测图
"""
import os, sys, json, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov3'))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import yaml

from config import cfg
from agent import Agent
from dataloader import get_noise, get_initial_states
from util import save_img, STATE_STOPPED_DIM
from yolov3.models.yolo import Model
from yolov3.utils.general import check_dataset, intersect_dicts, non_max_suppression
from yolov3.utils.torch_utils import select_device

FILTER_NAMES = ['E', 'G', 'CCM', 'Shr', 'NLM', 'T', 'Ct', 'S+', 'BW', 'W']


def draw_boxes(img_np, detections, class_names):
    """在 numpy RGB 图上画检测框, 返回 uint8 BGR"""
    out = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if detections is not None and len(detections):
        for *xyxy, conf, cls in detections.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_names.get(int(cls), int(cls))}: {conf:.2f}'
            cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--isp_weights', type=str, default='../pretrained/ckpt-lod-df-1.0.pth')
    parser.add_argument('--yolo_weights', type=str, default='../pretrained/yolov3.pt')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    args = parser.parse_args()

    device = select_device('0')
    print(f'Device: {device}')

    # ── 加载模型 ───────────────────────────────────
    # YOLOv3
    with open('yolov3/data/hyps/hyp.scratch-low.yaml') as f:
        hyp = yaml.safe_load(f)
    data_dict = check_dataset('yolov3/data/lod.yaml')
    ckpt_yolo = torch.load(args.yolo_weights, map_location='cpu')
    yolo = Model('yolov3/models/yolov3.yaml', ch=3, nc=data_dict['nc'],
                 anchors=hyp.get('anchors')).to(device)
    csd = intersect_dicts(ckpt_yolo['model'].float().state_dict(), yolo.state_dict(),
                          exclude=['anchor'])
    yolo.load_state_dict(csd, strict=False)
    yolo.eval()
    class_names = data_dict['names']  # dict {0:'person', 1:'bicycle', ...}
    print(f'YOLOv3: {data_dict["nc"]} classes')

    # Agent
    agent = Agent(cfg, shape=(6 + len(cfg.filters), 64, 64), device=str(device)).to(device)
    ckpt_isp = torch.load(args.isp_weights, map_location='cpu')
    agent.load_state_dict(ckpt_isp['agent_model'])
    agent.eval()
    print(f'Agent: {len(agent.filters)} filters')

    # ── 创建输出目录 ───────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.steps + 1):  # step-0 ~ step-5
        os.makedirs(os.path.join(args.output_dir, 'img_results', f'step-{i}'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'param_results'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detect_before'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detect_after'), exist_ok=True)

    # ── 搜集图片 ───────────────────────────────────
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = sorted([f for f in Path(args.image_dir).iterdir()
                        if f.suffix.lower() in exts])
    print(f'图片: {len(img_files)} 张')

    # ── 逐张处理 ───────────────────────────────────
    z_dim = 3 + len(cfg.filters) * 16
    num_state_dim = 3 + len(cfg.filters)
    filter_counter = Counter()
    records_lines = [','.join(FILTER_NAMES)]

    for img_path in tqdm(img_files, desc='ISP + YOLO'):
        torch.cuda.empty_cache()
        name = Path(img_path).stem

        # 加载图像
        img = Image.open(img_path).convert('RGB').resize((512, 512))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # ── 处理前 YOLO 检测 ────────────────────
        with torch.no_grad():
            pred_before = yolo(img_t)
            det_before = non_max_suppression(pred_before, args.conf_thres, 0.45, max_det=300)
        cv2.imwrite(os.path.join(args.output_dir, 'detect_before', f'{name}.jpg'),
                    draw_boxes(img_np, det_before[0], class_names))

        # ── 保存原始输入 ────────────────────────
        save_img(img_t, str(img_path),
                 os.path.join(args.output_dir, 'img_results', 'step-0'),
                 format='CHW', is_train=False)

        # ── ISP 推理 ────────────────────────────
        noises = torch.from_numpy(
            np.array([get_noise(1, 'uniform', z_dim) for _ in range(args.steps)])
        ).to(device)
        states = torch.from_numpy(
            get_initial_states(1, num_state_dim, len(cfg.filters))
        ).to(device)

        retouch = img_t
        filter_ids = []
        param_record = {'pipeline': []}

        with torch.no_grad():
            for i in range(args.steps):
                (retouch, new_states, _, _), debug_info, _ = agent(
                    (retouch, noises[i], states), 1.0)
                fid = debug_info['selected_filter_id'].item()
                fname = agent.filters[fid].get_short_name()
                filter_ids.append(str(fid))
                filter_counter[fname] += 1

                param_info = debug_info['filter_debug_info'][fid]['filter_parameters']
                param_record[fname] = param_info.detach().cpu().numpy().tolist()
                param_record['pipeline'].append(fid)

                save_img(retouch, str(img_path),
                         os.path.join(args.output_dir, 'img_results', f'step-{i + 1}'),
                         format='CHW', is_train=False)

                states = new_states
                if states[0][STATE_STOPPED_DIM] > 0:
                    break

        # ── 处理后 YOLO 检测 ────────────────────
        with torch.no_grad():
            pred_after = yolo(retouch)
            det_after = non_max_suppression(pred_after, args.conf_thres, 0.45, max_det=300)
        img_after_np = retouch[0].cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join(args.output_dir, 'detect_after', f'{name}.jpg'),
                    draw_boxes(img_after_np, det_after[0], class_names))

        # ── 记录 ────────────────────────────────
        records_lines.append(f'{img_path.name},{",".join(filter_ids)}')
        with open(os.path.join(args.output_dir, 'param_results', f'{name}.json'), 'w') as f:
            json.dump(param_record, f, indent=2, ensure_ascii=False)

    # ── records.txt ────────────────────────────────
    with open(os.path.join(args.output_dir, 'records.txt'), 'w') as f:
        f.write('\n'.join(records_lines) + '\n')

    # ── 终端汇总 ───────────────────────────────────
    print(f'\n{"=" * 50}')
    print(f'完成! {len(img_files)} 张图片')
    print(f'\n滤波器使用分布 ({len(img_files) * args.steps} 次选择):')
    for fname in FILTER_NAMES:
        n = filter_counter.get(fname, 0)
        pct = n / max(len(img_files) * args.steps, 1) * 100
        print(f'  {fname}: {n:4d} ({pct:5.1f}%) {"█" * int(pct / 2)}')
    print(f'\n输出: {args.output_dir}/')
    print(f'  img_results/step-0/ ~ step-{args.steps}/')
    print(f'  param_results/')
    print(f'  records.txt')
    print(f'  detect_before/  detect_after/')


if __name__ == '__main__':
    main()
