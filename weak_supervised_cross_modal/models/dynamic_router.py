"""
MAVD动态路由模块 - 基于动态MFMS搜索的伪标签生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class MAVDDynamicRouter(nn.Module):
    """
    MAVD动态路由模块
    
    实现功能：
    1. 动态模态特征匹配子空间（MFMS）搜索
    2. 噪声门控机制的伪标签生成
    3. 专家网络动态权重分配
    4. 迭代优化的属性发现
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        
        # 复用DUET的聚类提取器思想
        self.cluster_extractor = LinearExtractorCluster(config)
        
        # MAVD动态路由逻辑
        self.dynamic_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_experts),
        )
        
        # 噪声门控参数（可学习）
        self.noise_gate = nn.Parameter(torch.randn(config.num_experts))
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(config) for _ in range(config.num_experts)
        ])
        
        # 模态特征匹配子空间（MFMS）
        self.mfms_projectors = nn.ModuleDict({
            'color': nn.Linear(config.hidden_size, config.attr_dim),
            'material': nn.Linear(config.hidden_size, config.attr_dim),
            'shape': nn.Linear(config.hidden_size, config.attr_dim)
        })
        
        # 动态阈值学习
        self.dynamic_threshold = nn.Parameter(torch.tensor(0.5))
        
        # 伪标签质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_features: [B, hidden_size] 视觉特征
            
        Returns:
            pseudo_labels: [B, num_experts] 伪标签权重
            importance: [B, num_experts] 重要性权重
        """
        batch_size = visual_features.size(0)
        
        # 聚类特征提取
        clustered_feats, cluster_importance = self.cluster_extractor(visual_features)
        
        # 动态门控权重计算
        gate_logits = self.dynamic_gate(clustered_feats)  # [B, num_experts]
        
        # 添加噪声门控机制
        noise = torch.randn_like(gate_logits) * self.noise_scale
        noisy_gate_logits = gate_logits + noise + self.noise_gate.unsqueeze(0)
        
        # 动态阈值应用
        threshold_mask = torch.sigmoid(noisy_gate_logits - self.dynamic_threshold)
        
        # 生成伪标签权重
        pseudo_labels = torch.softmax(noisy_gate_logits, dim=-1) * threshold_mask
        
        # 专家网络处理
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(clustered_feats)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, hidden_size]
        
        # 加权聚合专家输出
        weighted_output = torch.sum(
            expert_outputs * pseudo_labels.unsqueeze(-1), 
            dim=1
        )  # [B, hidden_size]
        
        # 评估伪标签质量
        quality_scores = self.quality_assessor(weighted_output)  # [B, 1]
        
        # 重要性权重结合质量评估
        importance = cluster_importance * quality_scores
        
        return pseudo_labels, importance
    
    def mfms_search(self, features: torch.Tensor, 
                    attribute: str) -> torch.Tensor:
        """
        模态特征匹配子空间（MFMS）搜索
        
        Args:
            features: [B, hidden_size] 输入特征
            attribute: 目标属性名称
            
        Returns:
            projected_features: [B, attr_dim] 投影后特征
        """
        if attribute not in self.mfms_projectors:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        # 投影到属性特定子空间
        projected = self.mfms_projectors[attribute](features)
        
        # L2归一化
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected
    
    def iterative_optimization(self, features: torch.Tensor, 
                             num_iterations: int = 3) -> Dict[str, torch.Tensor]:
        """
        迭代优化的属性发现
        
        Args:
            features: [B, hidden_size] 输入特征
            num_iterations: 迭代次数
            
        Returns:
            optimized_attributes: 优化后的属性特征字典
        """
        current_features = features
        attribute_features = {}
        
        for iteration in range(num_iterations):
            # 为每个属性进行MFMS搜索
            for attr in ['color', 'material', 'shape']:
                # MFMS投影
                projected = self.mfms_search(current_features, attr)
                
                # 更新属性特征
                if attr not in attribute_features:
                    attribute_features[attr] = projected
                else:
                    # 指数移动平均更新
                    alpha = 0.7
                    attribute_features[attr] = (
                        alpha * attribute_features[attr] + 
                        (1 - alpha) * projected
                    )
            
            # 更新当前特征（融合所有属性信息）
            all_attr_feats = torch.cat(list(attribute_features.values()), dim=-1)
            update_proj = nn.Linear(
                all_attr_feats.size(-1), 
                features.size(-1)
            ).to(features.device)
            
            current_features = current_features + 0.1 * update_proj(all_attr_feats)
        
        return attribute_features
    
    def get_routing_statistics(self) -> Dict[str, torch.Tensor]:
        """获取路由统计信息（用于分析和调试）"""
        with torch.no_grad():
            stats = {
                'noise_gate_weights': self.noise_gate.clone(),
                'noise_scale': self.noise_scale.clone(),
                'dynamic_threshold': self.dynamic_threshold.clone(),
                'expert_utilization': torch.zeros(self.num_experts)
            }
        
        return stats

class LinearExtractorCluster(nn.Module):
    """
    线性特征提取聚类模块（基于DUET思想）
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 线性模式提取器
        self.pattern_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
        # 聚类中心学习
        self.cluster_centers = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size)
        )
        
        # 重要性权重网络
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size // 4, config.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [B, hidden_size] 输入特征
            
        Returns:
            clustered_features: [B, hidden_size] 聚类后特征
            importance_weights: [B, num_experts] 重要性权重
        """
        # 模式提取
        extracted_features = self.pattern_extractor(features)
        
        # 计算与聚类中心的距离
        distances = torch.cdist(
            extracted_features.unsqueeze(1),  # [B, 1, hidden_size]
            self.cluster_centers.unsqueeze(0)  # [1, num_experts, hidden_size]
        ).squeeze(1)  # [B, num_experts]
        
        # 软分配权重（距离越近权重越大）
        assignment_weights = F.softmax(-distances, dim=-1)  # [B, num_experts]
        
        # 加权聚合
        clustered_features = torch.sum(
            self.cluster_centers.unsqueeze(0) * assignment_weights.unsqueeze(-1),
            dim=1
        )  # [B, hidden_size]
        
        # 计算重要性权重
        importance_weights = self.importance_net(clustered_features)
        
        return clustered_features, importance_weights

class ExpertNetwork(nn.Module):
    """专家网络模块"""
    
    def __init__(self, config):
        super().__init__()
        
        self.expert_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 专家特异性参数
        self.expert_bias = nn.Parameter(torch.randn(config.hidden_size))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        专家网络前向传播
        
        Args:
            features: [B, hidden_size] 输入特征
            
        Returns:
            expert_output: [B, hidden_size] 专家输出
        """
        output = self.expert_layers(features)
        output = output + self.expert_bias  # 添加专家偏置
        
        return output

class NoisyGating(nn.Module):
    """噪声门控模块（用于增强路由的多样性）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 噪声生成网络
        self.noise_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size // 4, config.num_experts)
        )
        
        # 噪声强度控制
        self.noise_strength = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, features: torch.Tensor, 
                gate_logits: torch.Tensor) -> torch.Tensor:
        """
        添加自适应噪声到门控logits
        
        Args:
            features: [B, hidden_size] 输入特征
            gate_logits: [B, num_experts] 门控logits
            
        Returns:
            noisy_logits: [B, num_experts] 带噪声的门控logits
        """
        # 生成自适应噪声
        adaptive_noise = self.noise_generator(features)
        
        # 添加随机噪声
        random_noise = torch.randn_like(gate_logits) * self.noise_strength
        
        # 组合噪声
        total_noise = adaptive_noise + random_noise
        
        # 应用噪声
        noisy_logits = gate_logits + total_noise
        
        return noisy_logits 