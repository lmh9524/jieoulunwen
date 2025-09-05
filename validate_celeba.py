#!/usr/bin/env python3
"""
CelebA 测试集验证脚本
- 加载 best_model.pth
- 在 CelebA 测试集上评估平均损失与各属性组 Acc/F1（macro），并导出报告

使用示例：
python validate_celeba.py --checkpoint ./experiments/celeba_training_xxx/checkpoints/best_model.pth --device auto --save-report
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict

import torch
from torch.cuda.amp import autocast

# 项目路径（与 train_celeba.py 保持一致）
sys.path.append('./weak_supervised_cross_modal')

from config.base_config import get_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_dataset import CelebADatasetAdapter


def parse_args():
    parser = argparse.ArgumentParser(description='CelebA 测试集验证')
    parser.add_argument('--checkpoint', type=str, required=True, help='best_model.pth 路径')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖配置中的 batch_size')
    parser.add_argument('--num-workers', type=int, default=None, help='覆盖配置中的 num_workers')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='设备选择')
    parser.add_argument('--save-report', action='store_true', help='是否保存 JSON 报告到实验目录')
    return parser.parse_args()


def infer_device(arg_device: str) -> torch.device:
    if arg_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg_device)


def build_dataloader(config) -> Dict[str, torch.utils.data.DataLoader]:
    adapter = CelebADatasetAdapter(config)
    return adapter.get_dataloaders()


def load_model_and_state(config, checkpoint_path: str, device: torch.device):
    model = WeakSupervisedCrossModalAlignment(config)
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'未找到检查点: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        # 兼容直接保存 state_dict 的情况
        model.load_state_dict(ckpt, strict=False)

    return model, ckpt


def evaluate(model, dataloader, criterion, metrics: EvaluationMetrics, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    start = time.time()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss, _ = criterion(outputs, targets, epoch=0)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            if 'predictions' in outputs:
                metrics.update(outputs['predictions'], targets)

    elapsed = time.time() - start
    avg_loss = total_loss / max(1, total_samples)
    metric_dict = metrics.compute()

    return avg_loss, metric_dict, elapsed


def main():
    args = parse_args()

    # 加载配置（CelebA）
    config = get_config('CelebA')

    # 覆盖批大小/并行等
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    device = infer_device(args.device)
    config.device = device

    # 数据
    dataloaders = build_dataloader(config)
    test_loader = dataloaders['test']

    # 模型与损失
    model, ckpt = load_model_and_state(config, args.checkpoint, device)
    criterion = ComprehensiveLoss(config)
    metrics = EvaluationMetrics(config.num_classes)

    # 评估
    print('开始 CelebA 测试集验证...')
    if device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name()} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    avg_loss, metric_dict, elapsed = evaluate(model, test_loader, criterion, metrics, device)

    # 打印摘要
    print('\n===== CelebA 测试集评估结果 =====')
    print(f'平均损失: {avg_loss:.4f}')
    mean_acc = metric_dict.get('mean_accuracy', None)
    if mean_acc is not None:
        print(f'Mean Accuracy: {mean_acc:.4f}')
    # 每个属性组简表
    for attr in config.num_classes.keys():
        acc_key = f'{attr}_accuracy'
        f1_key = f'{attr}_f1'
        if acc_key in metric_dict:
            print(f'- {attr:16s} Acc={metric_dict[acc_key]:.4f}  F1={metric_dict.get(f1_key, 0.0):.4f}')
    print(f'用时: {elapsed/60:.1f} 分钟')

    # 选择性保存报告
    if args.save_report:
        # 推断实验目录（checkpoint 的上级目录的上级）
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        os.makedirs(exp_dir, exist_ok=True)
        report = {
            'dataset': 'CelebA',
            'checkpoint': os.path.abspath(args.checkpoint),
            'avg_loss': float(avg_loss),
            'metrics': metric_dict,
            'elapsed_minutes': elapsed / 60.0,
            'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        out_path = os.path.join(exp_dir, 'test_report.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'测试报告已保存: {out_path}')


if __name__ == '__main__':
    main() 