#!/usr/bin/env python3
"""
CelebA 测试集验证脚本 (适配实际数据集结构)
- 加载 best_model.pth
- 在 CelebA 测试集上评估 (使用实际数据路径)
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

# 项目路径
sys.path.append('./weak_supervised_cross_modal')

from config.base_config import get_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_dataset import CelebADatasetAdapter


def parse_args():
    parser = argparse.ArgumentParser(description='CelebA 测试集验证 (适配实际数据)')
    parser.add_argument('--checkpoint', type=str, required=True, help='best_model.pth 路径')
    parser.add_argument('--data-path', type=str, default='D:\\KKK\\data\\CelebA', help='CelebA数据集根目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载器进程数')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='设备选择')
    parser.add_argument('--save-report', action='store_true', help='是否保存 JSON 报告')
    return parser.parse_args()


def infer_device(arg_device: str) -> torch.device:
    if arg_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg_device)


def create_mock_eval_partition(data_path: str):
    """如果缺少评估分割文件，创建一个模拟的分割"""
    eval_dir = os.path.join(data_path, 'Eval')
    os.makedirs(eval_dir, exist_ok=True)
    
    partition_file = os.path.join(eval_dir, 'list_eval_partition.txt')
    if not os.path.exists(partition_file):
        print("创建模拟的数据集分割文件...")
        # 读取属性文件获取所有图像名
        attr_file = os.path.join(data_path, 'annotations', 'list_attr_celeba.txt')
        if os.path.exists(attr_file):
            with open(attr_file, 'r') as f:
                lines = f.readlines()
            
            # 跳过头部，获取图像名
            image_names = []
            for i, line in enumerate(lines[2:], 0):  # 跳过数量和列名行
                img_name = line.strip().split()[0]
                image_names.append(img_name)
            
            # 简单分割：前70%训练，20%验证，10%测试
            total = len(image_names)
            train_end = int(total * 0.7)
            val_end = int(total * 0.9)
            
            with open(partition_file, 'w') as f:
                for i, img_name in enumerate(image_names):
                    if i < train_end:
                        partition = 0  # train
                    elif i < val_end:
                        partition = 1  # val
                    else:
                        partition = 2  # test
                    f.write(f"{img_name} {partition}\n")
            
            print(f"已创建分割文件: {partition_file}")
            print(f"总样本: {total}, 训练: {train_end}, 验证: {val_end-train_end}, 测试: {total-val_end}")


def setup_celeba_structure(data_path: str):
    """适配CelebA数据集结构"""
    # 创建Anno目录链接到annotations
    anno_dir = os.path.join(data_path, 'Anno')
    annotations_dir = os.path.join(data_path, 'annotations')
    
    if not os.path.exists(anno_dir) and os.path.exists(annotations_dir):
        print("创建Anno目录软链接...")
        try:
            # Windows下创建目录符号链接
            os.system(f'mklink /D "{anno_dir}" "{annotations_dir}"')
        except:
            # 如果符号链接失败，复制文件
            import shutil
            shutil.copytree(annotations_dir, anno_dir, dirs_exist_ok=True)
    
    # 创建评估分割文件
    create_mock_eval_partition(data_path)


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
        model.load_state_dict(ckpt, strict=False)

    return model, ckpt


def evaluate(model, dataloader, criterion, metrics: EvaluationMetrics, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
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
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx+1}/{len(dataloader)}")

    elapsed = time.time() - start
    avg_loss = total_loss / max(1, total_samples)
    metric_dict = metrics.compute()

    return avg_loss, metric_dict, elapsed


def main():
    args = parse_args()

    # 检查数据集路径
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: CelebA数据集路径不存在: {args.data_path}")
        return

    print(f"✅ 使用CelebA数据集: {args.data_path}")
    
    # 适配数据集结构
    setup_celeba_structure(args.data_path)

    # 加载配置并修改数据路径
    config = get_config('CelebA')
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers

    device = infer_device(args.device)
    config.device = device

    # 数据加载器
    print("正在加载数据...")
    dataloaders = build_dataloader(config)
    test_loader = dataloaders['test']

    # 模型与损失
    print("正在加载模型...")
    model, ckpt = load_model_and_state(config, args.checkpoint, device)
    criterion = ComprehensiveLoss(config)
    metrics = EvaluationMetrics(config.num_classes)

    # 评估
    print('开始 CelebA 测试集验证...')
    if device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name()} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    avg_loss, metric_dict, elapsed = evaluate(model, test_loader, criterion, metrics, device)

    # 打印结果
    print('\n===== CelebA 测试集评估结果 =====')
    print(f'平均损失: {avg_loss:.4f}')
    mean_acc = metric_dict.get('mean_accuracy', None)
    if mean_acc is not None:
        print(f'Mean Accuracy: {mean_acc:.4f}')
    
    print('\n各属性组详细结果:')
    for attr in config.num_classes.keys():
        acc_key = f'{attr}_accuracy'
        f1_key = f'{attr}_f1'
        precision_key = f'{attr}_precision'
        recall_key = f'{attr}_recall'
        
        if acc_key in metric_dict:
            acc = metric_dict[acc_key]
            f1 = metric_dict.get(f1_key, 0.0)
            precision = metric_dict.get(precision_key, 0.0)
            recall = metric_dict.get(recall_key, 0.0)
            print(f'- {attr:16s} Acc={acc:.4f}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}')
    
    print(f'\n用时: {elapsed/60:.1f} 分钟')

    # 保存报告
    if args.save_report:
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        report = {
            'dataset': 'CelebA',
            'checkpoint': os.path.abspath(args.checkpoint),
            'data_path': args.data_path,
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