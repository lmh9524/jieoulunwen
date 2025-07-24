"""
视觉特征提取器
使用预训练的ResNet模型提取图像特征
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VisualEncoder(nn.Module):
    """视觉特征提取器"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 2048,
                 freeze_backbone: bool = False):
        """
        初始化视觉编码器
        
        Args:
            backbone: 骨干网络名称
            pretrained: 是否使用预训练权重
            feature_dim: 输出特征维度
            freeze_backbone: 是否冻结骨干网络
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # 创建骨干网络
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # 冻结骨干网络参数
        if freeze_backbone:
            self._freeze_backbone()
        
        # 获取骨干网络输出维度
        backbone_dim = self._get_backbone_dim()
        
        # 特征投影层
        if backbone_dim != feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
        
        logger.info(f"视觉编码器初始化完成: {backbone} -> {feature_dim}D")
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """创建骨干网络"""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # 移除最后的分类层
            model = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier = nn.Identity()
        elif backbone == 'vit_base':
            model = models.vit_b_16(pretrained=pretrained)
            model.heads = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        return model
    
    def _get_backbone_dim(self) -> int:
        """获取骨干网络输出维度"""
        if 'resnet' in self.backbone_name:
            if 'resnet50' in self.backbone_name or 'resnet101' in self.backbone_name or 'resnet152' in self.backbone_name:
                return 2048
            else:
                return 512
        elif 'efficientnet' in self.backbone_name:
            return 1280
        elif 'vit' in self.backbone_name:
            return 768
        else:
            # 默认值
            return 2048
    
    def _freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("骨干网络参数已冻结")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            特征张量 [batch_size, feature_dim]
        """
        # 骨干网络特征提取
        features = self.backbone(x)
        
        # 展平特征
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # 特征投影
        features = self.projection(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        """获取输出特征维度"""
        return self.feature_dim

class MultiScaleVisualEncoder(nn.Module):
    """多尺度视觉特征提取器"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 2048,
                 scales: list = [1, 2, 4]):
        """
        初始化多尺度视觉编码器
        
        Args:
            backbone: 骨干网络名称
            pretrained: 是否使用预训练权重
            feature_dim: 输出特征维度
            scales: 多尺度因子
        """
        super().__init__()
        
        self.scales = scales
        self.feature_dim = feature_dim
        
        # 创建多个尺度的编码器
        self.encoders = nn.ModuleList([
            VisualEncoder(backbone, pretrained, feature_dim // len(scales))
            for _ in scales
        ])
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        logger.info(f"多尺度视觉编码器初始化完成: {len(scales)} 个尺度")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            融合特征张量 [batch_size, feature_dim]
        """
        features = []
        
        for i, encoder in enumerate(self.encoders):
            scale = self.scales[i]
            
            # 缩放输入
            if scale != 1:
                h, w = x.size(2), x.size(3)
                scaled_x = nn.functional.interpolate(
                    x, size=(h // scale, w // scale), 
                    mode='bilinear', align_corners=False
                )
            else:
                scaled_x = x
            
            # 提取特征
            feat = encoder(scaled_x)
            features.append(feat)
        
        # 特征拼接
        fused_features = torch.cat(features, dim=1)
        
        # 特征融合
        output = self.fusion(fused_features)
        
        return output

class AttentionVisualEncoder(nn.Module):
    """注意力视觉特征提取器"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 2048,
                 num_heads: int = 8):
        """
        初始化注意力视觉编码器
        
        Args:
            backbone: 骨干网络名称
            pretrained: 是否使用预训练权重
            feature_dim: 输出特征维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # 基础编码器
        self.base_encoder = VisualEncoder(backbone, pretrained, feature_dim)
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        logger.info(f"注意力视觉编码器初始化完成: {num_heads} 个注意力头")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
            
        Returns:
            注意力增强特征张量 [batch_size, feature_dim]
        """
        # 基础特征提取
        features = self.base_encoder(x)  # [batch_size, feature_dim]
        
        # 添加序列维度用于注意力计算
        features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 自注意力
        attn_output, _ = self.self_attention(features, features, features)
        
        # 残差连接和层归一化
        features = self.layer_norm(features + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(features)
        features = self.layer_norm(features + ffn_output)
        
        # 移除序列维度
        output = features.squeeze(1)  # [batch_size, feature_dim]
        
        return output

def create_visual_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建视觉编码器
    
    Args:
        config: 配置字典
        
    Returns:
        视觉编码器模型
    """
    encoder_type = config.get('type', 'basic')
    
    if encoder_type == 'basic':
        return VisualEncoder(
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 2048),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    elif encoder_type == 'multiscale':
        return MultiScaleVisualEncoder(
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 2048),
            scales=config.get('scales', [1, 2, 4])
        )
    elif encoder_type == 'attention':
        return AttentionVisualEncoder(
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feature_dim=config.get('feature_dim', 2048),
            num_heads=config.get('num_heads', 8)
        )
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")

# 测试代码
if __name__ == '__main__':
    # 测试基础编码器
    encoder = VisualEncoder()
    x = torch.randn(4, 3, 224, 224)
    features = encoder(x)
    print(f"基础编码器输出形状: {features.shape}")
    
    # 测试多尺度编码器
    multi_encoder = MultiScaleVisualEncoder()
    features = multi_encoder(x)
    print(f"多尺度编码器输出形状: {features.shape}")
    
    # 测试注意力编码器
    attn_encoder = AttentionVisualEncoder()
    features = attn_encoder(x)
    print(f"注意力编码器输出形状: {features.shape}") 