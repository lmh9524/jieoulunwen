"""
AFANet频域解耦模块 - 基于频域分析的属性特征解耦
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class FrequencyDomainDecoupler(nn.Module):
    """
    AFANet频域解耦模块
    
    实现功能：
    1. 傅里叶变换分离高低频特征
    2. 高频特征捕获边缘、纹理信息（材质属性）
    3. 低频特征捕获主体区域信息（颜色、形状属性）
    4. 自适应频域分解与特征融合
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freq_cutoff = config.freq_cutoff
        
        # 高频特征处理分支（纹理、边缘）
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(3, config.high_freq_dim // 4, 3, 1, 1),
            nn.BatchNorm2d(config.high_freq_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.high_freq_dim // 4, config.high_freq_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(config.high_freq_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.high_freq_dim // 2, config.high_freq_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 低频特征处理分支（主体区域、颜色）
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(3, config.low_freq_dim // 4, 5, 1, 2),
            nn.BatchNorm2d(config.low_freq_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.low_freq_dim // 4, config.low_freq_dim // 2, 5, 1, 2),
            nn.BatchNorm2d(config.low_freq_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.low_freq_dim // 2, config.low_freq_dim, 5, 1, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 频域融合层
        total_dim = config.high_freq_dim + config.low_freq_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 自适应权重学习
        self.freq_attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 4, 2),  # 高频、低频权重
            nn.Softmax(dim=-1)
        )
        
        # 属性特定的频域掩码
        self.attribute_masks = nn.ParameterDict({
            'color': nn.Parameter(torch.randn(1, 1, config.image_size, config.image_size)),
            'material': nn.Parameter(torch.randn(1, 1, config.image_size, config.image_size)),
            'shape': nn.Parameter(torch.randn(1, 1, config.image_size, config.image_size))
        })
        
    def create_lowpass_mask(self, h: int, w: int, cutoff: float) -> torch.Tensor:
        """创建低通滤波器掩码"""
        # 创建频域坐标
        u = torch.arange(h, dtype=torch.float32).unsqueeze(1) - h // 2
        v = torch.arange(w, dtype=torch.float32).unsqueeze(0) - w // 2
        
        # 计算距离
        distance = torch.sqrt(u**2 + v**2)
        
        # 创建低通滤波器（高斯型）
        cutoff_freq = min(h, w) * cutoff
        mask = torch.exp(-(distance**2) / (2 * cutoff_freq**2))
        
        # 中心化
        mask = torch.fft.fftshift(mask)
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    def frequency_decomposition(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        频域分解
        
        Args:
            image: [B, C, H, W] 输入图像
            
        Returns:
            low_freq: [B, C, H, W] 低频分量
            high_freq: [B, C, H, W] 高频分量
        """
        B, C, H, W = image.shape
        
        # 傅里叶变换
        fft = torch.fft.fft2(image, dim=(-2, -1))
        
        # 创建低通滤波器掩码
        low_pass_mask = self.create_lowpass_mask(H, W, self.freq_cutoff)
        low_pass_mask = low_pass_mask.to(image.device)
        
        # 分离高低频
        low_freq_fft = fft * low_pass_mask
        high_freq_fft = fft * (1 - low_pass_mask)
        
        # 逆傅里叶变换
        low_freq = torch.fft.ifft2(low_freq_fft, dim=(-2, -1)).real
        high_freq = torch.fft.ifft2(high_freq_fft, dim=(-2, -1)).real
        
        return low_freq, high_freq
    
    def attribute_specific_filtering(self, image: torch.Tensor, 
                                   attribute: str) -> torch.Tensor:
        """
        属性特定的频域滤波
        
        Args:
            image: [B, C, H, W] 输入图像
            attribute: 属性名称 ('color', 'material', 'shape')
            
        Returns:
            filtered_image: [B, C, H, W] 滤波后图像
        """
        if attribute not in self.attribute_masks:
            return image
        
        # 获取属性特定掩码
        attr_mask = torch.sigmoid(self.attribute_masks[attribute])
        
        # 应用掩码
        filtered_image = image * attr_mask
        
        return filtered_image
    
    def forward(self, image: torch.Tensor, 
                attribute_focus: Optional[str] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image: [B, C, H, W] 输入图像
            attribute_focus: 可选的属性焦点
            
        Returns:
            features: [B, hidden_size] 解耦后的特征
        """
        # 属性特定滤波（如果指定）
        if attribute_focus:
            image = self.attribute_specific_filtering(image, attribute_focus)
        
        # 频域分解
        low_freq, high_freq = self.frequency_decomposition(image)
        
        # 特征提取
        high_feats = self.high_freq_branch(high_freq)  # [B, high_freq_dim, 1, 1]
        low_feats = self.low_freq_branch(low_freq)     # [B, low_freq_dim, 1, 1]
        
        # 展平特征
        high_feats = high_feats.flatten(1)  # [B, high_freq_dim]
        low_feats = low_feats.flatten(1)    # [B, low_freq_dim]
        
        # 特征拼接
        combined_feats = torch.cat([high_feats, low_feats], dim=1)  # [B, total_dim]
        
        # 自适应权重
        freq_weights = self.freq_attention(combined_feats)  # [B, 2]
        weighted_high = high_feats * freq_weights[:, 0:1]
        weighted_low = low_feats * freq_weights[:, 1:2]
        
        # 重新组合
        weighted_combined = torch.cat([weighted_high, weighted_low], dim=1)
        
        # 特征融合
        features = self.fusion_layer(weighted_combined)  # [B, hidden_size]
        
        return features
    
    def get_frequency_analysis(self, image: torch.Tensor) -> dict:
        """
        获取频域分析结果（用于可视化和调试）
        
        Args:
            image: [B, C, H, W] 输入图像
            
        Returns:
            analysis: 包含频域分析结果的字典
        """
        with torch.no_grad():
            low_freq, high_freq = self.frequency_decomposition(image)
            
            # 计算频域能量
            low_energy = torch.mean(low_freq**2, dim=[1, 2, 3])
            high_energy = torch.mean(high_freq**2, dim=[1, 2, 3])
            
            # 频域比例
            total_energy = low_energy + high_energy
            low_ratio = low_energy / (total_energy + 1e-8)
            high_ratio = high_energy / (total_energy + 1e-8)
            
            analysis = {
                'low_freq_image': low_freq,
                'high_freq_image': high_freq,
                'low_energy': low_energy,
                'high_energy': high_energy,
                'low_ratio': low_ratio,
                'high_ratio': high_ratio
            }
            
        return analysis

class AdaptiveFrequencyAttention(nn.Module):
    """自适应频域注意力模块"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 频域注意力网络
        self.freq_attention = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),  # 输入：原图+低频+高频
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),  # 输出：3个属性的注意力图
            nn.Sigmoid()
        )
        
    def forward(self, original: torch.Tensor, 
                low_freq: torch.Tensor, 
                high_freq: torch.Tensor) -> torch.Tensor:
        """
        生成自适应频域注意力
        
        Args:
            original: [B, C, H, W] 原始图像
            low_freq: [B, C, H, W] 低频分量
            high_freq: [B, C, H, W] 高频分量
            
        Returns:
            attention_maps: [B, 3, H, W] 三个属性的注意力图
        """
        # 拼接输入
        concat_input = torch.cat([original, low_freq, high_freq], dim=1)
        
        # 生成注意力图
        attention_maps = self.freq_attention(concat_input)
        
        return attention_maps 