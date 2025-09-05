"""
Focal Loss实现 - 处理类别不平衡问题
特别针对facial_hair、hair_basic等不平衡属性组
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    适用于类别不平衡的多分类问题
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        """
        Args:
            alpha: 类别权重张量 [num_classes] 或 None
            gamma: focusing参数，通常为2.0
            reduction: 'mean', 'sum', 'none'
            label_smoothing: 标签平滑参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] 预测logits
            targets: [N] 目标标签
            
        Returns:
            loss: focal loss值
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                 label_smoothing=self.label_smoothing)
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            logpt = -ce_loss
            focal_loss = at * (1 - pt) ** self.gamma * logpt
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss实现
    
    论文: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    基于有效样本数的类别平衡损失
    """
    
    def __init__(self, samples_per_class: torch.Tensor, beta: float = 0.9999, 
                 gamma: float = 2.0, loss_type: str = 'focal'):
        """
        Args:
            samples_per_class: 每个类别的样本数 [num_classes]
            beta: 重采样参数，通常为0.9或0.99或0.999
            gamma: focal loss的gamma参数
            loss_type: 'focal', 'sigmoid', 'softmax'
        """
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # 归一化
        
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            self.focal_loss = FocalLoss(alpha=weights, gamma=gamma)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'focal':
            return self.focal_loss(inputs, targets)
        elif self.loss_type == 'softmax':
            return F.cross_entropy(inputs, targets, weight=self.weights)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

def create_adaptive_loss(attr_name: str, num_classes: int, 
                        samples_per_class: Optional[torch.Tensor] = None) -> nn.Module:
    """
    为特定属性组创建自适应损失函数
    
    Args:
        attr_name: 属性组名称
        num_classes: 类别数
        samples_per_class: 每类样本数，如果未提供则使用默认配置
        
    Returns:
        loss_fn: 损失函数
    """
    # 针对已知不平衡的属性组使用Focal Loss
    imbalanced_attrs = ['facial_hair', 'hair_basic', 'accessories']
    
    if attr_name in imbalanced_attrs:
        if samples_per_class is not None:
            # 使用Class-Balanced Loss
            return ClassBalancedLoss(samples_per_class, beta=0.9999, gamma=2.0)
        else:
            # 使用标准Focal Loss
            return FocalLoss(gamma=2.0, label_smoothing=0.1)
    else:
        # 其他属性组使用带标签平滑的交叉熵
        return nn.CrossEntropyLoss(label_smoothing=0.05)

class AdaptiveLossManager:
    """自适应损失管理器"""
    
    def __init__(self, config):
        self.config = config
        self.loss_functions = {}
        self._setup_losses()
    
    def _setup_losses(self):
        """设置各属性组的损失函数"""
        for attr_name, num_classes in self.config.num_classes.items():
            self.loss_functions[attr_name] = create_adaptive_loss(attr_name, num_classes)
    
    def compute_loss(self, predictions: dict, targets: dict) -> dict:
        """
        计算各属性组的自适应损失
        
        Args:
            predictions: {attr_name: logits, ...}
            targets: {attr_name: labels, ...}
            
        Returns:
            losses: {attr_name: loss_value, ...}
        """
        losses = {}
        
        for attr_name in self.config.num_classes.keys():
            if attr_name in predictions and attr_name in targets:
                loss_fn = self.loss_functions[attr_name]
                loss_value = loss_fn(predictions[attr_name], targets[attr_name])
                losses[attr_name] = loss_value
        
        return losses
    
    def to(self, device):
        """移动损失函数到指定设备"""
        for loss_fn in self.loss_functions.values():
            if hasattr(loss_fn, 'to'):
                loss_fn.to(device)
        return self 