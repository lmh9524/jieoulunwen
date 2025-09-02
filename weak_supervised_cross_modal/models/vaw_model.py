"""
VAW模型 - 简化版本
"""

import torch
import torch.nn as nn
import torchvision.models as models

class VAWCrossModalModel(nn.Module):
    """VAW跨模态模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉编码器 (简化版ResNet)
        backbone = models.resnet18(pretrained=True)  # 使用更小的ResNet18
        self.visual_encoder = nn.Sequential(
            *list(backbone.children())[:-1],  # 移除最后的分类层
            nn.Flatten(),
            nn.Linear(512, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 属性分类器
        self.attribute_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.attr_dim)
        )
    
    def forward(self, images, attributes=None):
        # 视觉特征提取
        visual_features = self.visual_encoder(images)
        
        # 属性预测
        attr_predictions = self.attribute_classifier(visual_features)
        
        return {
            "attribute_predictions": attr_predictions,
            "visual_features": visual_features
        }

def create_vaw_model(config):
    """创建VAW模型"""
    model = VAWCrossModalModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"VAW模型创建完成:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model

