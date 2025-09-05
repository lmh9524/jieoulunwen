#!/usr/bin/env python3
"""
CelebA优化训练脚本 - Stage 1
基于测试集分析结果的改进版本
- 重新设计的属性分组
- 调整的损失权重
- 早停机制与学习率调度
- 增强的数据增强
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

# 导入优化配置和数据集
from config.celeba_optimized_config import get_optimized_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_optimized_dataset import CelebAOptimizedDatasetAdapter
from utils.logging_utils import setup_logging
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, cleanup_old_checkpoints

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights:
            model.load_state_dict(self.best_weights)

class CelebAOptimizedTrainer:
    """CelebA优化训练器"""
    
    def __init__(self, stage=1, data_path='D:\\KKK\\data\\CelebA'):
        """
        初始化优化训练器
        
        Args:
            stage: 训练阶段 (1, 2, 3)
            data_path: CelebA数据集路径
        """
        # 获取对应阶段的优化配置
        self.config = get_optimized_config(stage)
        self.stage = stage
        self.config.data_path = data_path
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config.device = self.device
        
        # 实验目录
        stage_suffix = f"_stage{stage}" if stage > 1 else ""
        self.experiment_name = f"celeba_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}{stage_suffix}"
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
            'val_acc': [],
            'learning_rates': []
        }
        
        # 早停和调度器
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=0.001
        )
        
        print("=" * 70)
        print(f"CelebA 优化训练器初始化 - Stage {stage}")
        print("=" * 70)
        print(f"设备: {self.device}")
        print(f"训练轮数: {self.config.num_epochs}")
        print(f"批处理大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"实验目录: {self.save_dir}")
        print(f"早停容忍: {self.config.early_stopping_patience} epochs")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def setup_data(self):
        """设置优化数据加载器"""
        print("\n设置优化数据加载器...")
        
        adapter = CelebAOptimizedDatasetAdapter(self.config)
        self.dataloaders = adapter.get_dataloaders()
        
        # 获取数据集信息
        train_size = len(self.dataloaders['train'].dataset)
        val_size = len(self.dataloaders['val'].dataset)
        test_size = len(self.dataloaders['test'].dataset)
        
        print(f"训练集: {train_size:,} 样本 (批次数: {len(self.dataloaders['train'])})")
        print(f"验证集: {val_size:,} 样本")
        print(f"测试集: {test_size:,} 样本")
        
        return True
    
    def setup_model(self):
        """设置优化模型"""
        print("\n设置优化模型...")
        
        # 创建模型
        self.model = WeakSupervisedCrossModalAlignment(self.config)
        self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 打印启用的模块
        enabled_modules = []
        if self.config.use_frequency_decoupling:
            enabled_modules.append("AFANet")
        if self.config.use_hierarchical_decomposition:
            enabled_modules.append("WINNER") 
        if self.config.use_dynamic_routing:
            enabled_modules.append("MAVD")
        if self.config.use_cmdl_regularization:
            enabled_modules.append("CMDL")
        
        print(f"启用模块: {enabled_modules if enabled_modules else ['仅基础分类']}")
        
        # 创建损失函数
        self.criterion = ComprehensiveLoss(self.config)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_reduce_factor,
            patience=self.config.lr_reduce_patience,
            verbose=True,
            min_lr=1e-7
        )
        
        # AMP混合精度
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        
        # OOM保护机制
        self.original_batch_size = self.config.batch_size
        self.oom_count = 0
        
        # 评估指标
        self.metrics = EvaluationMetrics(self.config.num_classes)
        
        return True
    
    def _handle_oom(self):
        """处理OOM异常"""
        self.oom_count += 1
        print(f"⚠️ GPU内存不足! 第{self.oom_count}次OOM")
        
        if self.oom_count <= 3:
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 减少batch_size
            new_batch_size = max(4, self.config.batch_size // 2)
            if new_batch_size != self.config.batch_size:
                print(f"🔧 自动调整batch_size: {self.config.batch_size} → {new_batch_size}")
                self.config.batch_size = new_batch_size
                
                # 重新创建数据加载器
                print("🔄 重新创建数据加载器...")
                self.setup_data()
        else:
            print("❌ 多次OOM，建议检查GPU内存或降低模型复杂度")
            raise RuntimeError("连续OOM超过3次，训练终止")
    
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
        
        start_time = time.time()
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
            try:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._handle_oom()
                    continue
                else:
                    raise e
            
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
            if batch_idx % 50 == 0:
                progress = 100. * batch_idx / len(self.dataloaders['train'])
                elapsed = time.time() - start_time
                eta = elapsed / max(1, batch_idx + 1) * (len(self.dataloaders['train']) - batch_idx - 1)
                print(f"  进度: {progress:.1f}%, 损失: {loss.item():.4f}, ETA: {eta/60:.1f}min")
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        avg_accuracy = {}
        for key in self.config.num_classes.keys():
            if total_predictions[key] > 0:
                avg_accuracy[key] = correct_predictions[key] / total_predictions[key]
        
        overall_accuracy = np.mean(list(avg_accuracy.values()))
        
        print(f"  训练损失: {avg_loss:.4f}")
        print(f"  训练准确率: {overall_accuracy:.4f}")
        print(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        return avg_loss, overall_accuracy
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        self.metrics.reset()
        
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
                
                # 更新评估指标
                if 'predictions' in outputs:
                    self.metrics.update(outputs['predictions'], targets)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        metric_results = self.metrics.compute()
        overall_accuracy = metric_results.get('mean_accuracy', 0.0)
        
        print(f"  验证损失: {avg_loss:.4f}")
        print(f"  验证准确率: {overall_accuracy:.4f}")
        
        # 打印各属性组性能
        print("  各属性组验证准确率:")
        for attr in self.config.num_classes.keys():
            acc = metric_results.get(f'{attr}_accuracy', 0.0)
            print(f"    {attr}: {acc:.4f}")
        
        return avg_loss, overall_accuracy, metric_results
    
    def save_checkpoint(self, epoch, is_best=False, metric_results=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'stage': self.stage,
            'metric_results': metric_results
        }
        
        # 保存当前检查点
        checkpoint_path = f"{self.save_dir}/checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = f"{self.save_dir}/checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  💾 保存最佳模型: best_model.pth")
        
        # 自动清理旧检查点，只保留最近3个
        try:
            cleanup_old_checkpoints(f"{self.save_dir}/checkpoints", keep_count=3)
        except Exception as e:
            print(f"  ⚠️ 清理旧检查点时出错: {e}")
    
    def train(self):
        """执行完整优化训练流程"""
        print(f"\n开始CelebA优化训练 - Stage {self.stage}...")
        
        # 设置数据和模型
        if not self.setup_data():
            return False
        
        if not self.setup_model():
            return False
        
        # 保存配置
        config_path = f"{self.save_dir}/config.json"
        with open(config_path, 'w') as f:
            config_dict = {
                'stage': self.stage,
                'dataset_name': self.config.dataset_name,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'image_size': self.config.image_size,
                'num_classes': self.config.num_classes,
                'loss_weights': self.config.loss_weights,
                'early_stopping_patience': self.config.early_stopping_patience,
                'enabled_modules': {
                    'frequency_decoupling': self.config.use_frequency_decoupling,
                    'hierarchical_decomposition': self.config.use_hierarchical_decomposition,
                    'dynamic_routing': self.config.use_dynamic_routing,
                    'cmdl_regularization': self.config.use_cmdl_regularization
                }
            }
            json.dump(config_dict, f, indent=2)
        
        # 训练循环
        start_time = time.time()
        
        print(f"🚀 开始Stage {self.stage}训练: 0 -> {self.config.num_epochs-1} epochs")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} - Stage {self.stage}")
            print(f"{'='*70}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, metric_results = self.validate_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best, metric_results)
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print(f"\n🛑 早停触发！在 Epoch {epoch+1} 停止训练")
                print(f"最佳验证损失: {self.early_stopping.best_loss:.4f}")
                if self.early_stopping.restore_best_weights:
                    self.early_stopping.restore_weights(self.model)
                    print("已恢复最佳权重")
                break
            
            # 打印摘要
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch+1} 摘要:")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  累计用时: {elapsed/60:.1f} 分钟")
            print(f"  早停计数: {self.early_stopping.counter}/{self.early_stopping.patience}")
        
        # 训练完成
        total_time = time.time() - start_time
        self.generate_training_report(total_time)
        
        return True
    
    def generate_training_report(self, total_time):
        """生成优化训练报告"""
        print("\n" + "="*70)
        print(f"CelebA Stage {self.stage} 优化训练完成报告")
        print("="*70)
        
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最终训练准确率: {self.training_history['train_acc'][-1]:.4f}")
        print(f"最终验证准确率: {self.training_history['val_acc'][-1]:.4f}")
        print(f"最终学习率: {self.training_history['learning_rates'][-1]:.2e}")
        
        # 保存训练历史
        history_path = f"{self.save_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存训练报告
        report = {
            "experiment_name": self.experiment_name,
            "stage": self.stage,
            "dataset": self.config.dataset_name,
            "total_epochs": len(self.training_history['train_loss']),
            "planned_epochs": self.config.num_epochs,
            "total_time_hours": total_time / 3600,
            "best_val_loss": self.best_val_loss,
            "final_train_acc": self.training_history['train_acc'][-1],
            "final_val_acc": self.training_history['val_acc'][-1],
            "final_lr": self.training_history['learning_rates'][-1],
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "early_stopped": len(self.training_history['train_loss']) < self.config.num_epochs,
            "config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_classes": self.config.num_classes,
                "loss_weights": self.config.loss_weights
            }
        }
        
        report_path = f"{self.save_dir}/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 训练报告已保存: {report_path}")
        print(f"📄 训练历史已保存: {history_path}")
        print(f"💾 最佳模型已保存: {self.save_dir}/checkpoints/best_model.pth")
        print("="*70)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CelebA优化训练')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], 
                       help='训练阶段 (1: 基础优化, 2: +轻量模块, 3: +完整模块)')
    parser.add_argument('--data-path', type=str, default='D:\\KKK\\data\\CelebA',
                       help='CelebA数据集路径')
    
    args = parser.parse_args()
    
    print("CelebA 优化训练启动")
    print("Copyright (c) 2024 - 弱监督解耦的跨模态属性对齐项目")
    
    # 检查数据集
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: CelebA数据集路径不存在: {args.data_path}")
        return
    
    print(f"✅ CelebA数据集检查通过: {args.data_path}")
    
    # 创建优化训练器并开始训练
    trainer = CelebAOptimizedTrainer(
        stage=args.stage,
        data_path=args.data_path
    )
    
    success = trainer.train()
    
    if success:
        print(f"\n🎉 CelebA Stage {args.stage} 优化训练成功完成!")
    else:
        print(f"\n❌ CelebA Stage {args.stage} 优化训练过程中出现错误")

if __name__ == "__main__":
    main() 