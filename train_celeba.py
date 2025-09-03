#!/usr/bin/env python3
"""
CelebA数据集训练脚本 - 弱监督解耦的跨模态属性对齐
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

# 添加项目路径
sys.path.append('./weak_supervised_cross_modal')

# 导入项目模块
from config.base_config import get_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_dataset import CelebADatasetAdapter
from utils.logging_utils import setup_logging
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

class CelebATrainer:
    """CelebA训练器"""
    
    def __init__(self, num_epochs=50, batch_size=16, learning_rate=1e-4):
        """
        初始化CelebA训练器
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批处理大小
            learning_rate: 学习率
        """
        # 获取配置
        self.config = get_config('CelebA')
        self.config.num_epochs = num_epochs
        self.config.batch_size = batch_size
        self.config.learning_rate = learning_rate
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config.device = self.device
        
        # 创建实验目录
        self.experiment_name = f"celeba_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = f"./experiments/{self.experiment_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print("=" * 60)
        print("CelebA 弱监督解耦训练器初始化")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练轮数: {num_epochs}")
        print(f"批处理大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"实验目录: {self.save_dir}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
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
        """设置模型"""
        print("\n设置模型...")
        
        # 创建模型
        self.model = WeakSupervisedCrossModalAlignment(self.config)
        self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 创建损失函数
        self.criterion = ComprehensiveLoss(self.config)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )
        
        # AMP混合精度
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        
        # 评估指标
        self.metrics = EvaluationMetrics(self.config.num_classes)
        
        return True
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = {}
        total_predictions = {}
        
        # 初始化准确率统计
        for key in self.config.num_classes.keys():
            correct_predictions[key] = 0
            total_predictions[key] = 0
        
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - 训练阶段")
        
        for batch_idx, (images, targets) in enumerate(self.dataloaders['train']):
            # 数据移到设备
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 前向传播
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(images)
                # 计算损失
                loss, loss_components = self.criterion(outputs, targets, epoch)
            
            # 反向传播 with AMP
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 计算准确率
            for key in self.config.num_classes.keys():
                if 'predictions' in outputs and key in outputs['predictions'] and key in targets:
                    logits = outputs['predictions'][key]
                    pred = torch.argmax(logits, dim=1)
                    correct = (pred == targets[key]).sum().item()
                    correct_predictions[key] += correct
                    total_predictions[key] += batch_size
            
            # 打印进度
            if batch_idx % 100 == 0:
                progress = 100. * batch_idx / len(self.dataloaders['train'])
                print(f"  进度: {progress:.1f}%, 损失: {loss.item():.4f}")
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
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
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = {}
        total_predictions = {}
        
        # 初始化准确率统计
        for key in self.config.num_classes.keys():
            correct_predictions[key] = 0
            total_predictions[key] = 0
        
        print(f"  验证阶段...")
        
        with torch.no_grad():
            for images, targets in self.dataloaders['val']:
                # 数据移到设备
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                with autocast(enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images)
                    # 计算损失
                    loss, _ = self.criterion(outputs, targets, epoch)
                
                # 统计
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 计算准确率
                for key in self.config.num_classes.keys():
                    if 'predictions' in outputs and key in outputs['predictions'] and key in targets:
                        logits = outputs['predictions'][key]
                        pred = torch.argmax(logits, dim=1)
                        correct = (pred == targets[key]).sum().item()
                        correct_predictions[key] += correct
                        total_predictions[key] += batch_size
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
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
    
    def train(self):
        """执行完整训练流程"""
        print("\n开始CelebA训练...")
        
        # 设置数据和模型
        if not self.setup_data():
            return False
        
        if not self.setup_model():
            return False
        
        # 保存配置
        config_path = f"{self.save_dir}/config.json"
        with open(config_path, 'w') as f:
            # 将配置转换为可序列化的字典
            config_dict = {
                'dataset_name': self.config.dataset_name,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'image_size': self.config.image_size,
                'num_classes': self.config.num_classes,
                'loss_weights': self.config.loss_weights
            }
            json.dump(config_dict, f, indent=2)
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
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
            print(f"  用时: {elapsed/60:.1f} 分钟")
        
        # 训练完成
        total_time = time.time() - start_time
        self.generate_training_report(total_time)
        
        return True
    
    def generate_training_report(self, total_time):
        """生成训练报告"""
        print("\n" + "="*60)
        print("CelebA 训练完成报告")
        print("="*60)
        
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最终训练准确率: {self.training_history['train_acc'][-1]:.4f}")
        print(f"最终验证准确率: {self.training_history['val_acc'][-1]:.4f}")
        
        # 保存训练历史
        history_path = f"{self.save_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存训练报告
        report = {
            "experiment_name": self.experiment_name,
            "dataset": "CelebA",
            "total_epochs": self.config.num_epochs,
            "total_time_hours": total_time / 3600,
            "best_val_loss": self.best_val_loss,
            "final_train_acc": self.training_history['train_acc'][-1],
            "final_val_acc": self.training_history['val_acc'][-1],
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_classes": self.config.num_classes
            }
        }
        
        report_path = f"{self.save_dir}/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 训练报告已保存: {report_path}")
        print(f"📄 训练历史已保存: {history_path}")
        print(f"💾 最佳模型已保存: {self.save_dir}/checkpoints/best_model.pth")
        print("="*60)

def main():
    """主函数"""
    print("CelebA 弱监督解耦训练启动")
    print("Copyright (c) 2024 - 弱监督解耦的跨模态属性对齐项目")
    
    # 检查CelebA数据集 (相对路径)
    celeba_path = ".."  # 相对于jieoulunwen-master目录
    if not os.path.exists(celeba_path):
        print(f"❌ 错误: CelebA数据集路径不存在: {celeba_path}")
        return
    
    if not os.path.exists(f"{celeba_path}/img_align_celeba"):
        print(f"❌ 错误: CelebA图像目录不存在: {os.path.abspath(celeba_path)}/img_align_celeba")
        return
        
    if not os.path.exists(f"{celeba_path}/Anno"):
        print(f"❌ 错误: CelebA标注目录不存在: {os.path.abspath(celeba_path)}/Anno")
        return
        
    if not os.path.exists(f"{celeba_path}/Eval"):
        print(f"❌ 错误: CelebA评估目录不存在: {os.path.abspath(celeba_path)}/Eval")
        return
    
    print(f"✅ CelebA数据集检查通过: {celeba_path}")
    
    # 创建训练器并开始训练
    trainer = CelebATrainer(
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4
    )
    
    success = trainer.train()
    
    if success:
        print("\n🎉 CelebA训练成功完成!")
    else:
        print("\n❌ CelebA训练过程中出现错误")

if __name__ == "__main__":
    main() 