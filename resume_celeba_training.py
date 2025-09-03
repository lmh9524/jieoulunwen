#!/usr/bin/env python3
"""
CelebA训练恢复脚本 - 从检查点继续训练
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import argparse

# 添加项目路径
sys.path.append('./weak_supervised_cross_modal')

# 导入项目模块
from config.base_config import get_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_dataset import CelebADatasetAdapter
from utils.logging_utils import setup_logging
from utils.checkpoint_utils import load_checkpoint

class CelebATrainerResume:
    """CelebA恢复训练器"""
    
    def __init__(self, checkpoint_path, start_epoch=None, total_epochs=50):
        """
        初始化恢复训练器
        
        Args:
            checkpoint_path: 检查点文件路径
            start_epoch: 指定开始轮次（可选）
            total_epochs: 总训练轮数
        """
        print("="*60)
        print("CelebA 恢复训练器初始化")
        print("="*60)
        
        # 加载检查点
        print(f"📂 加载检查点: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 加载检查点数据
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 从检查点恢复配置
        if 'config' in self.checkpoint:
            self.config = self.checkpoint['config']
        else:
            # 如果检查点中没有配置，使用默认配置
            self.config = get_config('CelebA')
        
        self.config.device = self.device
        self.config.num_epochs = total_epochs
        
        # 确定开始轮次
        self.start_epoch = start_epoch if start_epoch is not None else self.checkpoint.get('epoch', 0) + 1
        self.best_val_loss = self.checkpoint.get('best_val_loss', float('inf'))
        
        # 恢复训练历史
        self.training_history = self.checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        })
        
        # 创建新的实验目录
        self.experiment_name = f"celeba_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = f"./experiments/{self.experiment_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        print(f"开始轮次: {self.start_epoch}")
        print(f"总轮数: {total_epochs}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"新实验目录: {self.save_dir}")
        
        # 初始化AMP
        self.scaler = GradScaler()
    
    def setup_data(self):
        """设置数据加载器"""
        print("\n设置数据加载器...")
        
        adapter = CelebADatasetAdapter(self.config)
        self.dataloaders = adapter.get_dataloaders()
        
        # 获取数据集信息
        train_size = len(self.dataloaders['train'].dataset)
        val_size = len(self.dataloaders['val'].dataset)
        test_size = len(self.dataloaders['test'].dataset)
        
        print(f"训练集: {train_size:,} 样本")
        print(f"验证集: {val_size:,} 样本") 
        print(f"测试集: {test_size:,} 样本")
        
        return True
    
    def setup_model(self):
        """设置模型和优化器"""
        print("\n设置模型...")
        
        # 创建模型
        self.model = WeakSupervisedCrossModalAlignment(self.config)
        self.model.to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        print("✅ 模型权重已恢复")
        
        # 设置损失函数
        self.criterion = ComprehensiveLoss(self.config)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态已恢复")
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )
        
        # 恢复调度器状态
        if 'scheduler_state_dict' in self.checkpoint:
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
            print("✅ 学习率调度器状态已恢复")
        
        # 设置评估指标
        self.metrics = EvaluationMetrics(self.config.num_classes)
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return True
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = {key: 0 for key in self.config.num_classes.keys()}
        total_predictions = {key: 0 for key in self.config.num_classes.keys()}
        
        dataloader = self.dataloaders['train']
        total_batches = len(dataloader)
        
        print(f"Epoch {epoch+1}/{self.config.num_epochs} - 训练阶段")
        
        for batch_idx, batch in enumerate(dataloader):
            # 数据移到设备
            images = batch['image'].to(self.device)
            targets = {key: batch[key].to(self.device) for key in self.config.num_classes.keys()}
            
            # 前向传播（使用混合精度）
            with autocast():
                outputs = self.model(images, targets)
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            running_loss += loss.item()
            
            # 计算准确率
            for key in self.config.num_classes.keys():
                if key in outputs['predictions']:
                    predictions = outputs['predictions'][key].argmax(dim=1)
                    correct_predictions[key] += (predictions == targets[key]).sum().item()
                    total_predictions[key] += targets[key].size(0)
            
            # 打印进度
            if batch_idx % 100 == 0:
                progress = 100.0 * batch_idx / total_batches
                current_loss = running_loss / (batch_idx + 1)
                print(f"  进度: {progress:.1f}% 损失: {current_loss:.4f}")
        
        # 计算平均损失和准确率
        avg_loss = running_loss / total_batches
        avg_accuracy = {}
        for key in self.config.num_classes.keys():
            if total_predictions[key] > 0:
                avg_accuracy[key] = correct_predictions[key] / total_predictions[key]
        
        overall_accuracy = np.mean(list(avg_accuracy.values()))
        
        print(f"  训练损失: {avg_loss:.4f}")
        print(f"  训练准确率: {overall_accuracy:.4f}")
        
        return avg_loss, overall_accuracy
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = {key: 0 for key in self.config.num_classes.keys()}
        total_predictions = {key: 0 for key in self.config.num_classes.keys()}
        
        dataloader = self.dataloaders['val']
        total_batches = len(dataloader)
        
        print(f"Epoch {epoch+1}/{self.config.num_epochs} - 验证阶段")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 数据移到设备
                images = batch['image'].to(self.device)
                targets = {key: batch[key].to(self.device) for key in self.config.num_classes.keys()}
                
                # 前向传播（使用混合精度）
                with autocast():
                    outputs = self.model(images, targets)
                    loss = self.criterion(outputs, targets)
                
                # 统计
                running_loss += loss.item()
                
                # 计算准确率
                for key in self.config.num_classes.keys():
                    if key in outputs['predictions']:
                        predictions = outputs['predictions'][key].argmax(dim=1)
                        correct_predictions[key] += (predictions == targets[key]).sum().item()
                        total_predictions[key] += targets[key].size(0)
        
        # 计算平均损失和准确率
        avg_loss = running_loss / total_batches
        avg_accuracy = {}
        for key in self.config.num_classes.keys():
            if total_predictions[key] > 0:
                avg_accuracy[key] = correct_predictions[key] / total_predictions[key]
        
        overall_accuracy = np.mean(list(avg_accuracy.values()))
        
        print(f"  验证损失: {avg_loss:.4f}")
        print(f"  验证准确率: {overall_accuracy:.4f}")
        
        return avg_loss, overall_accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # 保存当前检查点
        checkpoint_path = f"{self.save_dir}/checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = f"{self.save_dir}/checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 保存最佳模型: {best_path}")
    
    def resume_train(self):
        """执行恢复训练"""
        print(f"\n🔄 从第 {self.start_epoch} 轮恢复CelebA训练...")
        
        # 设置数据和模型
        if not self.setup_data():
            return False
        
        if not self.setup_model():
            return False
        
        # 保存恢复配置
        config_path = f"{self.save_dir}/resume_config.json"
        with open(config_path, 'w') as f:
            config_dict = {
                'dataset_name': self.config.dataset_name,
                'num_epochs': self.config.num_epochs,
                'start_epoch': self.start_epoch,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'image_size': self.config.image_size,
                'num_classes': self.config.num_classes,
                'loss_weights': self.config.loss_weights,
                'resumed_from': self.checkpoint.get('epoch', 'unknown')
            }
            json.dump(config_dict, f, indent=2)
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} (恢复训练)")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 打印摘要
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch+1} 摘要:")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")
            print(f"  恢复训练用时: {elapsed/60:.1f} 分钟")
        
        # 训练完成
        total_time = time.time() - start_time
        self.generate_training_report(total_time)
        
        return True
    
    def generate_training_report(self, total_time):
        """生成训练报告"""
        print("\n" + "="*60)
        print("CelebA 恢复训练完成报告")
        print("="*60)
        
        print(f"恢复训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最终训练准确率: {self.training_history['train_acc'][-1]:.4f}")
        print(f"最终验证准确率: {self.training_history['val_acc'][-1]:.4f}")
        
        # 保存训练历史
        history_path = f"{self.save_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n📄 恢复训练历史已保存: {history_path}")
        print(f"💾 最佳模型已保存: {self.save_dir}/checkpoints/best_model.pth")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CelebA恢复训练脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点文件路径')
    parser.add_argument('--start_epoch', type=int, default=None, help='指定开始轮次（可选）')
    parser.add_argument('--total_epochs', type=int, default=50, help='总训练轮数')
    
    args = parser.parse_args()
    
    print("CelebA 恢复训练启动")
    print("Copyright (c) 2024 - 弱监督解耦的跨模态属性对齐项目")
    
    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 检查点文件不存在: {args.checkpoint}")
        return
    
    print(f"✅ 检查点文件检查通过: {args.checkpoint}")
    
    # 创建恢复训练器并开始训练
    trainer = CelebATrainerResume(
        checkpoint_path=args.checkpoint,
        start_epoch=args.start_epoch,
        total_epochs=args.total_epochs
    )
    
    success = trainer.resume_train()
    
    if success:
        print("\n🎉 CelebA恢复训练成功完成!")
    else:
        print("\n❌ CelebA恢复训练过程中出现错误")

if __name__ == "__main__":
    main() 