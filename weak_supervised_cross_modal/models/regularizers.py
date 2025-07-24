"""
CMDL轻量化正则化模块 - 基于互信息约束的属性解耦
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

class CMDLLightweightRegularizer(nn.Module):
    """
    CMDL轻量化正则化模块
    
    实现功能：
    1. 互信息估计与最小化
    2. 动态阈值MI估计器
    3. 属性特征解耦约束
    4. 轻量化计算优化
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_reg = config.cmdl_lambda
        self.hidden_size = config.hidden_size
        
        # 互信息估计网络
        self.mi_estimator = MutualInformationEstimator(config)
        
        # 动态阈值MI估计器
        self.dynamic_mi_estimator = DynamicThresholdMIEstimator(config)
        
        # 属性特定的正则化器
        self.attribute_regularizers = nn.ModuleDict({
            'color': AttributeRegularizer(config, 'color'),
            'material': AttributeRegularizer(config, 'material'),
            'shape': AttributeRegularizer(config, 'shape')
        })
        
        # 轻量化约束网络
        self.lightweight_constraint = LightweightConstraintNetwork(config)
        
        # 自适应权重学习
        self.adaptive_weights = nn.Parameter(torch.ones(3))  # 三个属性的权重
        
        # 温度参数（用于软化约束）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, attribute_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            attribute_features: 属性特征字典 {attr_name: [B, attr_dim]}
            
        Returns:
            total_loss: 总正则化损失
        """
        total_loss = 0.0
        mi_losses = {}
        
        # 计算属性间的互信息损失
        attribute_names = list(attribute_features.keys())
        for i, attr1 in enumerate(attribute_names):
            for j, attr2 in enumerate(attribute_names):
                if i < j:  # 避免重复计算
                    # 标准互信息损失
                    mi_loss = self.mutual_info_loss(
                        attribute_features[attr1], 
                        attribute_features[attr2]
                    )
                    
                    # 动态阈值MI损失
                    dynamic_mi_loss = self.dynamic_mi_estimator(
                        attribute_features[attr1],
                        attribute_features[attr2]
                    )
                    
                    # 组合损失
                    combined_loss = 0.7 * mi_loss + 0.3 * dynamic_mi_loss
                    mi_losses[f'{attr1}_{attr2}'] = combined_loss
                    
                    # 加权累加
                    weight = (self.adaptive_weights[i] + self.adaptive_weights[j]) / 2
                    total_loss += weight * combined_loss
        
        # 属性特定的正则化
        attr_reg_losses = {}
        for attr_name, attr_features in attribute_features.items():
            if attr_name in self.attribute_regularizers:
                attr_reg_loss = self.attribute_regularizers[attr_name](attr_features)
                attr_reg_losses[attr_name] = attr_reg_loss
                total_loss += 0.1 * attr_reg_loss
        
        # 轻量化约束
        lightweight_loss = self.lightweight_constraint(attribute_features)
        total_loss += 0.05 * lightweight_loss
        
        # 应用温度缩放
        total_loss = total_loss / self.temperature
        
        # 最终权重
        final_loss = self.lambda_reg * total_loss
        
        return final_loss
    
    def mutual_info_loss(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征间的互信息损失
        
        Args:
            feature1: [B, dim1] 第一个特征
            feature2: [B, dim2] 第二个特征
            
        Returns:
            mi_loss: 互信息损失（负值，用于最小化）
        """
        # 使用MINE（Mutual Information Neural Estimation）方法
        mi_score = self.mi_estimator(feature1, feature2)
        
        # 返回负互信息（最小化互信息实现解耦）
        return -mi_score
    
    def get_regularization_stats(self) -> Dict[str, torch.Tensor]:
        """获取正则化统计信息"""
        with torch.no_grad():
            stats = {
                'lambda_reg': self.lambda_reg,
                'adaptive_weights': self.adaptive_weights.clone(),
                'temperature': self.temperature.clone(),
                'mi_estimator_params': sum(p.numel() for p in self.mi_estimator.parameters()),
                'total_params': sum(p.numel() for p in self.parameters())
            }
        
        return stats

class MutualInformationEstimator(nn.Module):
    """互信息估计器（基于MINE方法）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 导入维度适配器
        from .dimension_adapter import DimensionAdapter
        
        # 维度适配器（用于处理不同的输入维度）
        self.input_adapter = None  # 将在forward中动态创建
        
        # 统计网络T(x,y)
        self.statistics_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.mi_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.mi_hidden_dim, config.mi_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.mi_hidden_dim // 2, 1)
        )
        
        # 指数移动平均（用于稳定训练）
        self.register_buffer('ema_et', torch.tensor(1.0))
        self.ema_decay = 0.99
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        估计互信息 I(X;Y)
        
        Args:
            x: [B, dim_x] 第一个变量
            y: [B, dim_y] 第二个变量
            
        Returns:
            mi_estimate: 互信息估计值
        """
        batch_size = x.size(0)
        
        # 联合分布：(x, y)
        joint_xy = torch.cat([x, y], dim=-1)  # [B, dim_x + dim_y]
        
        # 边际分布：(x, y') 其中y'是y的随机排列
        y_shuffle = y[torch.randperm(batch_size)]
        marginal_xy = torch.cat([x, y_shuffle], dim=-1)
        
        # 使用维度适配器处理维度不匹配
        expected_input_dim = self.config.hidden_size * 2
        actual_input_dim = joint_xy.size(-1)
        
        if actual_input_dim != expected_input_dim:
            # 动态创建适配器
            if self.input_adapter is None or self.input_adapter.get_input_dim() != actual_input_dim:
                from .dimension_adapter import DimensionAdapter
                self.input_adapter = DimensionAdapter(actual_input_dim, expected_input_dim)
                self.input_adapter = self.input_adapter.to(joint_xy.device)
            
            # 应用适配器
            joint_xy = self.input_adapter(joint_xy)
            marginal_xy = self.input_adapter(marginal_xy)
        
        # 计算统计量
        t_joint = self.statistics_network(joint_xy)  # [B, 1]
        t_marginal = self.statistics_network(marginal_xy)  # [B, 1]
        
        # MINE下界估计
        # I(X;Y) >= E_P[T(x,y)] - log(E_Q[exp(T(x,y'))])
        
        # 联合项
        joint_term = t_joint.mean()
        
        # 边际项（使用指数移动平均稳定训练）
        exp_t_marginal = torch.exp(t_marginal)
        if self.training:
            # 更新指数移动平均
            self.ema_et = self.ema_decay * self.ema_et + (1 - self.ema_decay) * exp_t_marginal.mean()
            marginal_term = torch.log(self.ema_et + 1e-8)
        else:
            marginal_term = torch.log(exp_t_marginal.mean() + 1e-8)
        
        # 互信息估计
        mi_estimate = joint_term - marginal_term
        
        return mi_estimate

class DynamicThresholdMIEstimator(nn.Module):
    """动态阈值MI估计器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 维度适配器
        self.threshold_adapter = None
        self.mi_adapter = None
        
        # 阈值预测网络
        self.threshold_predictor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()  # 输出0-1的阈值
        )
        
        # 自适应MI估计器
        self.adaptive_mi_estimator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.mi_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.mi_hidden_dim, 1)
        )
        
        # 特征相似度计算器
        self.similarity_calculator = FeatureSimilarityCalculator(config)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        动态阈值MI估计
        
        Args:
            x: [B, dim_x] 第一个特征
            y: [B, dim_y] 第二个特征
            
        Returns:
            dynamic_mi_loss: 动态MI损失
        """
        # 计算特征相似度
        similarity = self.similarity_calculator(x, y)
        
        # 预测动态阈值
        combined_features = torch.cat([x, y], dim=-1)
        
        # 使用维度适配器处理维度不匹配
        expected_input_dim = self.config.hidden_size * 2
        actual_input_dim = combined_features.size(-1)
        
        if actual_input_dim != expected_input_dim:
            # 动态创建适配器
            if self.threshold_adapter is None or self.threshold_adapter.get_input_dim() != actual_input_dim:
                from .dimension_adapter import DimensionAdapter
                self.threshold_adapter = DimensionAdapter(actual_input_dim, expected_input_dim)
                self.threshold_adapter = self.threshold_adapter.to(combined_features.device)
                
                self.mi_adapter = DimensionAdapter(actual_input_dim, expected_input_dim)
                self.mi_adapter = self.mi_adapter.to(combined_features.device)
            
            # 应用适配器
            threshold_input = self.threshold_adapter(combined_features)
            mi_input = self.mi_adapter(combined_features)
        else:
            threshold_input = combined_features
            mi_input = combined_features
        
        threshold = self.threshold_predictor(threshold_input)  # [B, 1]
        
        # 自适应MI估计
        mi_estimate = self.adaptive_mi_estimator(mi_input)  # [B, 1]
        
        # 应用动态阈值
        # 如果相似度高于阈值，则增加MI损失
        threshold_mask = (similarity > threshold.squeeze(-1)).float().unsqueeze(-1)
        
        # 动态损失
        dynamic_loss = threshold_mask * mi_estimate + (1 - threshold_mask) * 0.1 * mi_estimate
        
        return dynamic_loss.mean()

class AttributeRegularizer(nn.Module):
    """属性特定的正则化器"""
    
    def __init__(self, config, attribute_name: str):
        super().__init__()
        self.config = config
        self.attribute_name = attribute_name
        
        # 属性特定的约束网络
        self.constraint_network = nn.Sequential(
            nn.Linear(config.attr_dim, config.attr_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.attr_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 稀疏性约束
        self.sparsity_weight = nn.Parameter(torch.tensor(0.01))
        
        # 属性中心
        self.register_buffer(
            'attribute_center', 
            torch.zeros(config.attr_dim)
        )
        
    def forward(self, attribute_features: torch.Tensor) -> torch.Tensor:
        """
        属性特定正则化
        
        Args:
            attribute_features: [B, attr_dim] 属性特征
            
        Returns:
            reg_loss: 正则化损失
        """
        # 约束强度预测
        constraint_strength = self.constraint_network(attribute_features)  # [B, 1]
        
        # 中心化损失（鼓励特征聚集）
        center_loss = F.mse_loss(
            attribute_features.mean(dim=0), 
            self.attribute_center
        )
        
        # 稀疏性损失
        sparsity_loss = torch.mean(torch.abs(attribute_features)) * self.sparsity_weight
        
        # 一致性损失（同一属性的特征应该相似）
        consistency_loss = torch.var(attribute_features, dim=0).mean()
        
        # 加权组合
        total_loss = (
            constraint_strength.mean() * center_loss +
            sparsity_loss +
            0.1 * consistency_loss
        )
        
        return total_loss

class LightweightConstraintNetwork(nn.Module):
    """轻量化约束网络"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 轻量化网络（使用分组线性层模拟深度可分离卷积）
        num_groups = max(1, config.attr_dim // 4)
        group_size = config.attr_dim // num_groups
        
        # 创建分组线性层
        self.group_layers = nn.ModuleList([
            nn.Linear(group_size, group_size) for _ in range(num_groups)
        ])
        
        self.depthwise_constraint = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(config.attr_dim, config.attr_dim//4),
            nn.ReLU(inplace=True),
            nn.Linear(config.attr_dim//4, 1)
        )
        
        self.num_groups = num_groups
        self.group_size = group_size
        
        # 计算复杂度约束
        self.complexity_penalty = nn.Parameter(torch.tensor(0.001))
        
    def forward(self, attribute_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        轻量化约束计算
        
        Args:
            attribute_features: 属性特征字典
            
        Returns:
            lightweight_loss: 轻量化损失
        """
        total_constraint = 0.0
        
        for attr_name, attr_feats in attribute_features.items():
            # 应用分组线性层（如果特征维度匹配）
            if attr_feats.size(-1) == self.num_groups * self.group_size:
                # 重塑为分组形式
                batch_size = attr_feats.size(0)
                grouped_feats = attr_feats.view(batch_size, self.num_groups, self.group_size)
                
                # 应用每个分组的线性层
                group_outputs = []
                for i, group_layer in enumerate(self.group_layers):
                    group_output = group_layer(grouped_feats[:, i, :])  # [B, group_size]
                    group_outputs.append(group_output)
                
                # 重新组合
                processed_feats = torch.cat(group_outputs, dim=-1)  # [B, attr_dim]
            else:
                # 如果维度不匹配，直接使用原始特征
                processed_feats = attr_feats
            
            # 计算约束强度
            constraint = self.depthwise_constraint(processed_feats)  # [B, 1]
            total_constraint += constraint.mean()
        
        # 复杂度惩罚
        complexity_loss = self.complexity_penalty * total_constraint
        
        return complexity_loss

class FeatureSimilarityCalculator(nn.Module):
    """特征相似度计算器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 维度适配器
        self.input_adapter = None
        
        # 相似度度量网络
        self.similarity_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算特征相似度
        
        Args:
            x: [B, dim_x] 第一个特征
            y: [B, dim_y] 第二个特征
            
        Returns:
            similarity: [B] 相似度分数
        """
        # 确保特征维度一致
        if x.size(-1) != y.size(-1):
            min_dim = min(x.size(-1), y.size(-1))
            x = x[:, :min_dim]
            y = y[:, :min_dim]
        
        # 拼接特征
        combined = torch.cat([x, y], dim=-1)
        
        # 使用维度适配器处理维度不匹配
        expected_input_dim = self.config.hidden_size * 2
        actual_input_dim = combined.size(-1)
        
        if actual_input_dim != expected_input_dim:
            # 动态创建适配器
            if self.input_adapter is None or self.input_adapter.get_input_dim() != actual_input_dim:
                from .dimension_adapter import DimensionAdapter
                self.input_adapter = DimensionAdapter(actual_input_dim, expected_input_dim)
                self.input_adapter = self.input_adapter.to(combined.device)
            
            # 应用适配器
            combined = self.input_adapter(combined)
        
        # 计算相似度
        similarity = self.similarity_network(combined).squeeze(-1)  # [B]
        
        return similarity

class MILossScheduler:
    """互信息损失调度器"""
    
    def __init__(self, initial_lambda: float = 0.1, 
                 decay_rate: float = 0.95,
                 min_lambda: float = 0.01):
        self.initial_lambda = initial_lambda
        self.current_lambda = initial_lambda
        self.decay_rate = decay_rate
        self.min_lambda = min_lambda
        self.step_count = 0
        
    def step(self, mi_values: Dict[str, float]) -> float:
        """
        根据MI值调整lambda
        
        Args:
            mi_values: 当前的MI值字典
            
        Returns:
            new_lambda: 新的lambda值
        """
        self.step_count += 1
        
        # 计算平均MI值
        avg_mi = np.mean(list(mi_values.values()))
        
        # 如果MI值过高，增加正则化强度
        if avg_mi > 0.5:
            self.current_lambda = min(self.current_lambda * 1.1, 1.0)
        # 如果MI值过低，减少正则化强度
        elif avg_mi < 0.1:
            self.current_lambda = max(self.current_lambda * self.decay_rate, self.min_lambda)
        
        return self.current_lambda
    
    def get_current_lambda(self) -> float:
        """获取当前lambda值"""
        return self.current_lambda 