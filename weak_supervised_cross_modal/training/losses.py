"""
综合损失函数 - 集成所有创新点的损失计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

class ComprehensiveLoss(nn.Module):
    """
    综合损失函数
    
    集成所有创新点的损失：
    1. 分类损失（属性预测）
    2. CAL对比对齐损失
    3. MAVD动态路由损失
    4. WINNER层级一致性损失
    5. CMDL正则化损失
    6. 频域解耦损失
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_weights = config.loss_weights
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(config)
        
        # 层级一致性损失
        self.hierarchy_loss = HierarchyConsistencyLoss(config)
        
        # 频域多样性损失
        self.frequency_loss = FrequencyDiversityLoss(config)
        
        # 动态权重调整
        self.dynamic_weights = DynamicWeightAdjuster(config)
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算综合损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标标签字典
            epoch: 当前训练轮次
            
        Returns:
            total_loss: 总损失
            loss_components: 各组件损失字典
        """
        loss_components = {}
        total_loss = 0.0
        
        # 1. 属性分类损失
        if 'predictions' in outputs:
            cls_loss = self._compute_classification_loss(outputs['predictions'], targets)
            loss_components['classification_loss'] = cls_loss
            total_loss += self.loss_weights.get('classification', 1.0) * cls_loss
        
        # 2. CAL对比对齐损失
        if 'raw_attributes' in outputs:
            cal_loss = self.contrastive_loss(outputs['raw_attributes'], targets)
            loss_components['contrastive_alignment_loss'] = cal_loss
            total_loss += self.loss_weights.get('cal', 0.05) * cal_loss
        
        # 3. MAVD动态路由损失
        if 'pseudo_labels' in outputs and 'importance_weights' in outputs:
            mavd_loss = self._compute_mavd_loss(
                outputs['pseudo_labels'], 
                outputs['importance_weights'],
                targets
            )
            loss_components['mavd_loss'] = mavd_loss
            total_loss += self.loss_weights.get('mavd', 0.03) * mavd_loss
        
        # 4. WINNER层级一致性损失
        if 'hierarchical_features' in outputs:
            hier_loss = self.hierarchy_loss(outputs['hierarchical_features'])
            loss_components['hierarchy_loss'] = hier_loss
            total_loss += self.loss_weights.get('hierarchy', 0.01) * hier_loss
        
        # 5. CMDL正则化损失
        if 'regularization_loss' in outputs:
            reg_loss = outputs['regularization_loss']
            loss_components['regularization_loss'] = reg_loss
            total_loss += self.loss_weights.get('reg', 0.1) * reg_loss
        
        # 6. 频域解耦损失
        if 'frequency_features' in outputs:
            freq_loss = self.frequency_loss(outputs['frequency_features'])
            loss_components['frequency_loss'] = freq_loss
            total_loss += self.loss_weights.get('frequency', 0.005) * freq_loss
        
        # 7. 图表示损失
        if 'graph_representation' in outputs:
            graph_loss = self._compute_graph_loss(outputs['graph_representation'], targets)
            loss_components['graph_loss'] = graph_loss
            total_loss += self.loss_weights.get('graph', 0.02) * graph_loss
        
        # 动态权重调整
        adjusted_weights = self.dynamic_weights.update_weights(loss_components, epoch)
        
        # 重新计算加权总损失
        weighted_total_loss = 0.0
        for loss_name, loss_value in loss_components.items():
            if loss_name != 'total_loss':
                weight = adjusted_weights.get(loss_name, 1.0)
                weighted_total_loss += weight * loss_value
        
        loss_components['total_loss'] = weighted_total_loss
        loss_components['dynamic_weights'] = adjusted_weights
        
        return weighted_total_loss, loss_components
    
    def _compute_classification_loss(self, predictions: Dict[str, torch.Tensor], 
                                   targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算分类损失"""
        total_cls_loss = 0.0
        num_attributes = 0
        
        for attr_name, pred_logits in predictions.items():
            if attr_name in targets:
                attr_loss = self.ce_loss(pred_logits, targets[attr_name])
                
                # 属性特定权重
                attr_weight = self.loss_weights.get(f'{attr_name}_cls', 1.0)
                total_cls_loss += attr_weight * attr_loss
                num_attributes += 1
        
        return total_cls_loss / max(1, num_attributes)
    
    def _compute_mavd_loss(self, pseudo_labels: torch.Tensor, 
                          importance_weights: torch.Tensor,
                          targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算MAVD动态路由损失"""
        mavd_loss = 0.0
        
        # 1. 伪标签质量损失
        if 'pseudo_quality' in targets:
            quality_loss = self.mse_loss(
                importance_weights.mean(dim=-1), 
                targets['pseudo_quality']
            )
            mavd_loss += quality_loss
        
        # 2. 路由多样性损失
        # 鼓励专家网络的多样化使用
        expert_usage = torch.mean(pseudo_labels, dim=0)  # [num_experts]
        uniform_distribution = torch.ones_like(expert_usage) / expert_usage.size(0)
        diversity_loss = self.kl_loss(
            F.log_softmax(expert_usage, dim=0).unsqueeze(0),
            uniform_distribution.unsqueeze(0)
        )
        mavd_loss += 0.1 * diversity_loss
        
        # 3. 重要性权重稀疏性损失
        sparsity_loss = torch.mean(torch.abs(importance_weights))
        mavd_loss += 0.01 * sparsity_loss
        
        return mavd_loss
    
    def _compute_graph_loss(self, graph_representation: torch.Tensor,
                           targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算图表示损失"""
        graph_loss = 0.0
        
        # 图表示的正则化
        # 1. 图嵌入的L2正则化
        l2_reg = torch.mean(graph_representation**2)
        graph_loss += 0.01 * l2_reg
        
        # 2. 如果有图标签，计算监督损失
        if 'graph_labels' in targets:
            # 这里可以添加图级别的监督信号
            pass
        
        return graph_loss

class ContrastiveLoss(nn.Module):
    """CAL对比对齐损失"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = 0.07
        
    def forward(self, attributes: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算对比对齐损失
        
        Args:
            attributes: 属性特征字典
            targets: 目标标签字典
            
        Returns:
            contrastive_loss: 对比损失
        """
        total_loss = 0.0
        num_pairs = 0
        
        # 属性间对比学习
        attr_names = list(attributes.keys())
        for i, attr1 in enumerate(attr_names):
            for j, attr2 in enumerate(attr_names):
                if i < j and 'aligned' not in attr1 and 'aligned' not in attr2:
                    # 获取属性特征
                    feat1 = attributes[attr1]
                    feat2 = attributes[attr2]
                    
                    # 池化到固定维度
                    if feat1.dim() > 2:
                        feat1 = feat1.mean(dim=1)
                    if feat2.dim() > 2:
                        feat2 = feat2.mean(dim=1)
                    
                    # 计算对比损失
                    contrastive = self._compute_pairwise_contrastive_loss(
                        feat1, feat2, attr1, attr2, targets
                    )
                    total_loss += contrastive
                    num_pairs += 1
        
        return total_loss / max(1, num_pairs)
    
    def _compute_pairwise_contrastive_loss(self, feat1: torch.Tensor, feat2: torch.Tensor,
                                         attr1: str, attr2: str,
                                         targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算两个属性间的对比损失"""
        batch_size = feat1.size(0)
        
        # 归一化特征
        feat1_norm = F.normalize(feat1, p=2, dim=-1)
        feat2_norm = F.normalize(feat2, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.mm(feat1_norm, feat2_norm.t()) / self.temperature
        
        # 创建标签（如果可用）
        if attr1 in targets and attr2 in targets:
            labels1 = targets[attr1]
            labels2 = targets[attr2]
            
            # 正样本：相同类别
            positive_mask = (labels1.unsqueeze(1) == labels2.unsqueeze(0)).float()
            negative_mask = 1 - positive_mask
        else:
            # 如果没有标签，使用特征相似度生成伪标签
            with torch.no_grad():
                sim_threshold = 0.5
                positive_mask = (similarity > sim_threshold).float()
                negative_mask = 1 - positive_mask
        
        # InfoNCE损失
        pos_loss = -torch.log(torch.exp(similarity) * positive_mask + 1e-8)
        neg_loss = -torch.log(1 - torch.exp(similarity) * negative_mask + 1e-8)
        
        contrastive_loss = (pos_loss + neg_loss).mean()
        
        return contrastive_loss

class HierarchyConsistencyLoss(nn.Module):
    """WINNER层级一致性损失"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算层级一致性损失
        
        Args:
            hierarchical_features: 各层级特征列表
            
        Returns:
            hierarchy_loss: 层级一致性损失
        """
        if len(hierarchical_features) < 2:
            return torch.tensor(0.0, device=hierarchical_features[0].device)
        
        total_loss = 0.0
        
        # 1. 层级间平滑性约束
        smoothness_loss = 0.0
        for i in range(len(hierarchical_features) - 1):
            current_level = hierarchical_features[i].mean(dim=1)  # [B, hidden_size]
            next_level = hierarchical_features[i + 1].mean(dim=1)
            
            # L2距离
            level_diff = F.mse_loss(current_level, next_level)
            smoothness_loss += level_diff
        
        total_loss += smoothness_loss / (len(hierarchical_features) - 1)
        
        # 2. 层级多样性约束
        diversity_loss = 0.0
        for i in range(len(hierarchical_features)):
            for j in range(i + 1, len(hierarchical_features)):
                feat_i = hierarchical_features[i].mean(dim=1)
                feat_j = hierarchical_features[j].mean(dim=1)
                
                # 余弦相似性
                similarity = F.cosine_similarity(feat_i, feat_j, dim=-1).mean()
                
                # 鼓励不同层级学习不同特征
                diversity_penalty = torch.relu(similarity - 0.3)  # 阈值0.3
                diversity_loss += diversity_penalty
        
        num_pairs = len(hierarchical_features) * (len(hierarchical_features) - 1) // 2
        total_loss += 0.1 * diversity_loss / max(1, num_pairs)
        
        # 3. 层级结构约束
        structure_loss = 0.0
        for i, level_feat in enumerate(hierarchical_features):
            # 高层级应该有更全局的表示（更小的方差）
            level_variance = torch.var(level_feat, dim=1).mean()
            expected_variance = 1.0 / (i + 1)  # 层级越高，方差越小
            
            structure_penalty = F.mse_loss(
                level_variance, 
                torch.tensor(expected_variance, device=level_feat.device)
            )
            structure_loss += structure_penalty
        
        total_loss += 0.05 * structure_loss / len(hierarchical_features)
        
        return total_loss

class FrequencyDiversityLoss(nn.Module):
    """AFANet频域多样性损失"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, frequency_features: torch.Tensor) -> torch.Tensor:
        """
        计算频域多样性损失
        
        Args:
            frequency_features: [B, 1, hidden_size] 频域特征
            
        Returns:
            frequency_loss: 频域多样性损失
        """
        if frequency_features.dim() == 3:
            frequency_features = frequency_features.squeeze(1)  # [B, hidden_size]
        
        batch_size = frequency_features.size(0)
        
        # 1. 特征多样性损失
        normalized_features = F.normalize(frequency_features, p=2, dim=-1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # 减少非对角线元素（鼓励多样性）
        mask = torch.eye(batch_size, device=frequency_features.device)
        diversity_loss = torch.sum(similarity_matrix * (1 - mask)) / (batch_size * (batch_size - 1))
        
        # 2. 频域特征分布约束
        # 鼓励特征分布的均匀性
        feature_mean = torch.mean(frequency_features, dim=0)
        feature_std = torch.std(frequency_features, dim=0)
        
        # 均值应该接近0
        mean_penalty = torch.mean(feature_mean**2)
        
        # 标准差应该适中
        std_penalty = torch.mean((feature_std - 1.0)**2)
        
        distribution_loss = mean_penalty + 0.1 * std_penalty
        
        total_loss = diversity_loss + 0.1 * distribution_loss
        
        return total_loss

class DynamicWeightAdjuster:
    """动态权重调整器"""
    
    def __init__(self, config):
        self.config = config
        self.loss_history = {}
        self.weight_history = {}
        self.adaptation_rate = 0.1
        
    def update_weights(self, loss_components: Dict[str, torch.Tensor], 
                      epoch: int) -> Dict[str, float]:
        """
        根据损失历史动态调整权重
        
        Args:
            loss_components: 当前损失组件
            epoch: 当前轮次
            
        Returns:
            adjusted_weights: 调整后的权重
        """
        adjusted_weights = {}
        
        for loss_name, loss_value in loss_components.items():
            if loss_name == 'total_loss' or loss_name == 'dynamic_weights':
                continue
                
            loss_val = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            
            # 更新损失历史
            if loss_name not in self.loss_history:
                self.loss_history[loss_name] = []
            self.loss_history[loss_name].append(loss_val)
            
            # 计算损失趋势
            if len(self.loss_history[loss_name]) > 5:
                recent_losses = self.loss_history[loss_name][-5:]
                loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
                
                # 获取当前权重
                current_weight = self.config.loss_weights.get(loss_name, 1.0)
                
                # 动态调整
                if loss_trend > 0:  # 损失上升，增加权重
                    new_weight = current_weight * (1 + self.adaptation_rate)
                else:  # 损失下降，可以减少权重
                    new_weight = current_weight * (1 - self.adaptation_rate * 0.5)
                
                # 限制权重范围
                new_weight = max(0.001, min(2.0, new_weight))
                adjusted_weights[loss_name] = new_weight
            else:
                adjusted_weights[loss_name] = self.config.loss_weights.get(loss_name, 1.0)
        
        return adjusted_weights
    
    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取权重统计信息"""
        stats = {}
        for loss_name, history in self.loss_history.items():
            if len(history) > 0:
                stats[loss_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'trend': (history[-1] - history[0]) / len(history) if len(history) > 1 else 0.0
                }
        return stats 