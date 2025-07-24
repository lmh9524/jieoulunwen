"""
WINNER层级分解模块 - 基于层级注意力的结构化语义生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple

class WINNERHierarchicalDecomposer(nn.Module):
    """
    WINNER层级分解模块
    
    实现功能：
    1. 多层级特征分解树构建
    2. 结构化注意力机制
    3. 属性关系图生成
    4. 虚假关联缓解
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels
        self.hidden_size = config.hidden_size
        
        # 复用VLN-DUET的注意力机制
        self.attention_layers = nn.ModuleList([
            HierarchicalAttention(config, level=i) 
            for i in range(config.num_levels)
        ])
        
        # 层级嵌入
        self.level_embeddings = nn.Embedding(config.num_levels, config.hidden_size)
        
        # 属性关系图生成器
        self.graph_generator = AttributeGraphGenerator(config)
        
        # 结构化语义编码器
        self.semantic_encoder = StructuredSemanticEncoder(config)
        
        # 虚假关联检测器
        self.spurious_detector = SpuriousCorrelationDetector(config)
        
        # 层级融合网络
        self.level_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * config.num_levels, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
        # 自适应层级权重
        self.level_attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_levels),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor, 
                level_ids: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            level_ids: [B, seq_len] 可选的层级ID
            
        Returns:
            hierarchical_feats: List of [B, seq_len, hidden_size] 各层级特征
            graph_repr: [B, graph_dim] 属性关系图表示
        """
        batch_size, seq_len, _ = features.shape
        hierarchical_feats = []
        
        current_features = features
        
        # 逐层级处理
        for level, attention_layer in enumerate(self.attention_layers):
            # 添加层级嵌入
            level_embed = self.level_embeddings(
                torch.tensor(level, device=features.device)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            level_features = current_features + level_embed
            
            # 层级注意力处理
            attended_features = attention_layer(
                level_features, 
                level=level,
                mask=None
            )
            
            hierarchical_feats.append(attended_features)
            
            # 更新当前特征（残差连接）
            current_features = attended_features + current_features
        
        # 虚假关联检测与缓解
        clean_features = self.spurious_detector(hierarchical_feats)
        
        # 生成属性关系图
        graph_repr = self.graph_generator(clean_features)
        
        return hierarchical_feats, graph_repr
    
    def build_decomposition_tree(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        构建分解树结构
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            
        Returns:
            tree_structure: 分解树的字典表示
        """
        tree_structure = {}
        
        # 根节点（全局特征）
        tree_structure['root'] = features.mean(dim=1)  # [B, hidden_size]
        
        # 中间层级（属性组合）
        for level in range(self.num_levels - 1):
            level_features = self.attention_layers[level](features)
            tree_structure[f'level_{level}'] = level_features.mean(dim=1)
        
        # 叶子节点（单一属性）
        leaf_features = self.attention_layers[-1](features)
        
        # 分解为不同属性
        attr_dim = leaf_features.size(-1) // 3  # 假设三个属性
        tree_structure['color'] = leaf_features[:, :, :attr_dim].mean(dim=1)
        tree_structure['material'] = leaf_features[:, :, attr_dim:2*attr_dim].mean(dim=1)
        tree_structure['shape'] = leaf_features[:, :, 2*attr_dim:].mean(dim=1)
        
        return tree_structure
    
    def compute_structure_loss(self, hierarchical_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        计算结构化损失（确保层级一致性）
        
        Args:
            hierarchical_feats: 各层级特征列表
            
        Returns:
            structure_loss: 结构化损失
        """
        structure_loss = 0.0
        
        # 层级一致性损失
        for i in range(len(hierarchical_feats) - 1):
            current_level = hierarchical_feats[i].mean(dim=1)  # [B, hidden_size]
            next_level = hierarchical_feats[i + 1].mean(dim=1)
            
            # 计算层级间的相似性损失
            consistency_loss = F.mse_loss(current_level, next_level)
            structure_loss += consistency_loss
        
        # 多样性损失（确保不同层级学到不同特征）
        for i in range(len(hierarchical_feats)):
            for j in range(i + 1, len(hierarchical_feats)):
                feat_i = hierarchical_feats[i].mean(dim=1)
                feat_j = hierarchical_feats[j].mean(dim=1)
                
                # 余弦相似性
                similarity = F.cosine_similarity(feat_i, feat_j, dim=-1).mean()
                
                # 多样性损失（相似性应该较低）
                diversity_loss = torch.relu(similarity - 0.3)  # 阈值0.3
                structure_loss += 0.1 * diversity_loss
        
        return structure_loss

class HierarchicalAttention(nn.Module):
    """层级注意力模块"""
    
    def __init__(self, config, level: int):
        super().__init__()
        self.config = config
        self.level = level
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 查询、键、值投影
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 输出投影
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 层级特定的位置编码
        self.level_position_encoding = PositionalEncoding(
            config.hidden_size, level_specific=True, level=level
        )
        
        # 结构化掩码生成器
        self.mask_generator = StructuredMaskGenerator(config, level)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, features: torch.Tensor, 
                level: int, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        层级注意力前向传播
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            level: 当前层级
            mask: 可选的注意力掩码
            
        Returns:
            output: [B, seq_len, hidden_size] 输出特征
        """
        batch_size, seq_len, _ = features.shape
        
        # 添加层级特定的位置编码
        features_with_pos = self.level_position_encoding(features)
        
        # 计算查询、键、值
        Q = self.query(features_with_pos)  # [B, seq_len, hidden_size]
        K = self.key(features_with_pos)
        V = self.value(features_with_pos)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 生成结构化掩码
        if mask is None:
            mask = self.mask_generator(features, level)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            attention_scores = attention_scores + mask
        
        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        
        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # 输出投影
        output = self.output(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + features)
        
        return output

class AttributeGraphGenerator(nn.Module):
    """属性关系图生成器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 图节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.graph_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
        # 图边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.graph_dim * 2, config.graph_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.graph_dim, 1),
            nn.Sigmoid()
        )
        
        # 属性节点嵌入
        self.attribute_embeddings = nn.Embedding(3, config.graph_dim)  # color, material, shape
        
    def forward(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """
        生成属性关系图
        
        Args:
            hierarchical_features: 各层级特征列表
            
        Returns:
            graph_representation: [B, graph_dim] 图表示
        """
        # 提取属性节点特征
        attr_features = []
        for i, attr in enumerate(['color', 'material', 'shape']):
            # 使用最后一层的特征作为属性特征
            attr_feat = hierarchical_features[-1].mean(dim=1)  # [B, hidden_size]
            attr_feat = self.node_encoder(attr_feat)  # [B, graph_dim]
            
            # 添加属性嵌入
            attr_embed = self.attribute_embeddings(
                torch.tensor(i, device=attr_feat.device)
            ).unsqueeze(0)  # [1, graph_dim]
            
            attr_feat = attr_feat + attr_embed
            attr_features.append(attr_feat)
        
        # 计算属性间关系
        graph_edges = []
        for i in range(len(attr_features)):
            for j in range(i + 1, len(attr_features)):
                # 拼接两个属性特征
                edge_input = torch.cat([attr_features[i], attr_features[j]], dim=-1)
                # 预测边权重
                edge_weight = self.edge_predictor(edge_input)  # [B, 1]
                graph_edges.append(edge_weight)
        
        # 聚合图表示
        node_repr = torch.stack(attr_features, dim=1).mean(dim=1)  # [B, graph_dim]
        edge_repr = torch.cat(graph_edges, dim=-1).mean(dim=-1, keepdim=True)  # [B, 1]
        
        # 组合节点和边信息
        graph_representation = node_repr + edge_repr * node_repr
        
        return graph_representation

class StructuredSemanticEncoder(nn.Module):
    """结构化语义编码器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 语义结构网络
        self.semantic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout)
            ) for _ in range(3)  # 三层语义编码
        ])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """结构化语义编码"""
        current_features = features
        
        for layer in self.semantic_layers:
            current_features = layer(current_features) + current_features
        
        return current_features

class SpuriousCorrelationDetector(nn.Module):
    """虚假关联检测器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 关联检测网络
        self.correlation_detector = nn.Sequential(
            nn.Linear(config.hidden_size * config.num_levels, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.num_levels),
            nn.Sigmoid()  # 输出每个层级的可信度
        )
        
    def forward(self, hierarchical_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        检测并缓解虚假关联
        
        Args:
            hierarchical_features: 各层级特征列表
            
        Returns:
            clean_features: 清理后的特征列表
        """
        # 拼接所有层级特征
        concat_features = torch.cat([
            feat.mean(dim=1) for feat in hierarchical_features
        ], dim=-1)  # [B, hidden_size * num_levels]
        
        # 检测可信度
        reliability_scores = self.correlation_detector(concat_features)  # [B, num_levels]
        
        # 应用可信度权重
        clean_features = []
        for i, feat in enumerate(hierarchical_features):
            weight = reliability_scores[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            clean_feat = feat * weight
            clean_features.append(clean_feat)
        
        return clean_features

class PositionalEncoding(nn.Module):
    """层级特定的位置编码"""
    
    def __init__(self, hidden_size: int, level_specific: bool = True, level: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.level_specific = level_specific
        self.level = level
        
        if level_specific:
            # 为每个层级学习不同的位置编码
            self.position_embeddings = nn.Parameter(
                torch.randn(1000, hidden_size) * 0.02  # 支持最大1000的序列长度
            )
        else:
            # 使用标准的正弦位置编码
            self.register_buffer('position_embeddings', self._get_sinusoidal_encoding(1000, hidden_size))
    
    def _get_sinusoidal_encoding(self, max_len: int, hidden_size: int) -> torch.Tensor:
        """生成正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(math.log(10000.0) / hidden_size))
        
        pos_encoding = torch.zeros(max_len, hidden_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            
        Returns:
            encoded_features: [B, seq_len, hidden_size] 编码后特征
        """
        seq_len = features.size(1)
        
        if self.level_specific:
            # 层级特定编码
            pos_encoding = self.position_embeddings[:seq_len].unsqueeze(0)
            # 为不同层级添加不同的缩放因子
            scale_factor = 1.0 + 0.1 * self.level
            pos_encoding = pos_encoding * scale_factor
        else:
            pos_encoding = self.position_embeddings[:seq_len].unsqueeze(0)
        
        return features + pos_encoding

class StructuredMaskGenerator(nn.Module):
    """结构化掩码生成器"""
    
    def __init__(self, config, level: int):
        super().__init__()
        self.config = config
        self.level = level
        
        # 掩码生成网络
        self.mask_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
    def forward(self, features: torch.Tensor, level: int) -> torch.Tensor:
        """
        生成结构化注意力掩码
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            level: 当前层级
            
        Returns:
            mask: [B, num_heads, seq_len, seq_len] 注意力掩码
        """
        batch_size, seq_len, _ = features.shape
        num_heads = self.config.num_attention_heads
        
        # 基于特征生成掩码权重
        mask_weights = self.mask_generator(features)  # [B, seq_len, 1]
        
        # 创建结构化掩码
        mask = torch.full((batch_size, seq_len, seq_len), -float('inf'), device=features.device)
        
        # 根据层级调整掩码模式
        if level == 0:
            # 最低层级：局部注意力
            for i in range(seq_len):
                start = max(0, i - 2)
                end = min(seq_len, i + 3)
                mask[:, i, start:end] = 0.0  # 允许注意力的位置设为0
        elif level == 1:
            # 中间层级：中等范围注意力
            for i in range(seq_len):
                start = max(0, i - 5)
                end = min(seq_len, i + 6)
                mask[:, i, start:end] = 0.0
        else:
            # 最高层级：全局注意力
            mask.fill_(0.0)  # 允许所有位置的注意力
        
        # 扩展到多头维度: [B, 1, seq_len, seq_len] -> [B, num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        
        return mask