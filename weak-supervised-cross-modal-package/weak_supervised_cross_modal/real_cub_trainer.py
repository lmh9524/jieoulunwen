#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实CUB数据集跨模态训练器
真正使用图像数据进行跨模态属性对齐训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import logging
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from train_cub import CUBDataset
from utils.config_loader import ConfigLoader
from utils.decorators import error_handler, performance_monitor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealCrossModalModel(nn.Module):
    """真实的跨模态模型 - 使用真实图像特征"""
    
    def __init__(self, num_classes=200, num_attributes=312, hidden_dim=512, dropout=0.1):
        super().__init__()
        
        # 真实的视觉编码器 - 使用预训练ResNet
        self.visual_backbone = models.resnet50(pretrained=True)
        # 移除最后的分类层
        self.visual_backbone = nn.Sequential(*list(self.visual_backbone.children())[:-1])
        
        # 冻结部分预训练参数
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        
        # 解冻最后几层
        for param in self.visual_backbone[-2:].parameters():
            param.requires_grad = True
        
        # 视觉特征投影
        self.visual_projector = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 属性编码器
        self.attribute_encoder = nn.Sequential(
            nn.Linear(num_attributes, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 跨模态融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout * 0.5)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images, attributes):
        """前向传播 - 真实使用图像数据"""
        batch_size = images.size(0)
        
        # 真实的视觉特征提取
        with torch.no_grad():
            visual_features = self.visual_backbone(images)  # 使用真实图像！
        visual_features = visual_features.view(batch_size, -1)
        visual_features = self.visual_projector(visual_features)
        
        # 属性特征
        attribute_features = self.attribute_encoder(attributes)
        
        # 跨模态融合
        combined_features = torch.cat([visual_features, attribute_features], dim=1)
        
        # 注意力权重
        attention_weights = self.attention(combined_features)
        
        # 加权融合
        weighted_visual = visual_features * attention_weights
        weighted_attribute = attribute_features * (1 - attention_weights)
        final_features = torch.cat([weighted_visual, weighted_attribute], dim=1)
        
        # 最终融合和分类
        fused_features = self.fusion(final_features)
        logits = self.classifier(fused_features)
        
        return {
            'predictions': {'species': logits},
            'logits': logits,
            'visual_features': visual_features,
            'attribute_features': attribute_features,
            'attention_weights': attention_weights
        }

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info('Restored best model weights')
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()

class RealCUBTrainer:
    """真实CUB数据训练器"""
    
    def __init__(self, config_path: str):
        self.config = ConfigLoader(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建保存目录
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_early_stopping()
        
    def _setup_data(self):
        """设置数据加载器"""
        # 数据变换 - 适合ResNet的预处理
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 数据集
        data_root = self.config.get('data.root_path', '../data/CUB_200_2011/CUB_200_2011')
        
        self.train_dataset = CUBDataset(
            data_root=data_root,
            split='train',
            transform=train_transform,
            use_attributes=True
        )
        
        self.test_dataset = CUBDataset(
            data_root=data_root,
            split='test', 
            transform=test_transform,
            use_attributes=True
        )
        
        # 数据加载器
        batch_size = self.config.get('data.batch_size', 16)  # 减小batch size以适应真实图像处理
        num_workers = self.config.get('data.num_workers', 4)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"训练集: {len(self.train_dataset)} 样本")
        logger.info(f"测试集: {len(self.test_dataset)} 样本")
        
    def _setup_model(self):
        """设置模型"""
        dropout = self.config.get('model.dropout', 0.1)
        
        self.model = RealCrossModalModel(
            num_classes=200,
            num_attributes=312,
            hidden_dim=512,
            dropout=dropout
        ).to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        
    def _setup_training(self):
        """设置训练组件"""
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器 - 使用不同的学习率
        lr = self.config.get('training.learning_rate', 1e-4)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        
        # 分组参数 - 预训练部分使用更小的学习率
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'visual_backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.1},  # 预训练部分使用更小学习率
            {'params': other_params, 'lr': lr}
        ], weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler_type = self.config.get('training.scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = self.config.get('training.num_epochs', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        else:
            step_size = self.config.get('training.step_size', 15)
            gamma = self.config.get('training.gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            
        logger.info(f"使用学习率调度器: {scheduler_type}")
        
    def _setup_early_stopping(self):
        """设置早停机制"""
        patience = self.config.get('training.early_stopping.patience', 10)
        min_delta = self.config.get('training.early_stopping.min_delta', 0.001)
        
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True
        )
        
        logger.info(f"早停机制: patience={patience}, min_delta={min_delta}")
    
    @error_handler(default_return=None)
    @performance_monitor()
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            class_ids = batch['class_id'].to(self.device)
            attributes = batch['attributes'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播 - 使用真实图像
            outputs = self.model(images, attributes)
            loss = self.criterion(outputs['logits'], class_ids)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += class_ids.size(0)
            correct += (predicted == class_ids).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    @error_handler(default_return={})
    @performance_monitor()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                class_ids = batch['class_id'].to(self.device)
                attributes = batch['attributes'].to(self.device)
                
                outputs = self.model(images, attributes)
                loss = self.criterion(outputs['logits'], class_ids)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += class_ids.size(0)
                correct += (predicted == class_ids).sum().item()
        
        return {
            'loss': total_loss / len(self.test_loader),
            'accuracy': 100. * correct / total
        }
    
    def save_model(self, filename: str, epoch: int, is_best: bool = False):
        """保存模型"""
        save_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': dict(self.config.config),
            'is_best': is_best
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.checkpoint_dir / 'real_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, num_epochs: int = 50):
        """完整训练流程"""
        logger.info(f"开始真实图像跨模态训练，共 {num_epochs} 个epochs")
        logger.info(f"设备: {self.device}")
        logger.info(f"使用真实ResNet50特征提取器")
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 评估
            val_metrics = self.evaluate()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # 更新学习率
            self.scheduler.step()
            
            # 检查是否是最佳模型
            is_best = val_metrics['accuracy'] > best_acc
            if is_best:
                best_acc = val_metrics['accuracy']
                self.save_model('best_real_cub_model.pth', epoch, is_best=True)
            
            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                self.save_model(f'real_checkpoint_epoch_{epoch+1}.pth', epoch)
            
            # 早停检查
            self.early_stopping(val_metrics['accuracy'], self.model)
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}% - "
                f"LR: {current_lr:.6f} - "
                f"Best: {best_acc:.2f}%"
            )
            
            # 保存训练历史
            if (epoch + 1) % 5 == 0:
                self.save_history()
            
            # 早停检查
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 保存最后一个模型
        self.save_model('last_real_cub_model.pth', epoch)
        self.save_history()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {total_time/3600:.2f}小时")
        logger.info(f"最佳验证准确率: {best_acc:.2f}%")
        
        return best_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='真实CUB数据集跨模态训练')
    parser.add_argument('--config', type=str, default='configs/cub_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = RealCUBTrainer(args.config)
    
    # 开始训练
    best_acc = trainer.train(num_epochs=args.epochs)
    
    print(f"\n🎉 真实跨模态训练完成！最佳验证准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 