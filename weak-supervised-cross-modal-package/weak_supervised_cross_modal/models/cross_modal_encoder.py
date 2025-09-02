"""
跨模态编码器 - 基于VLN-DUET架构的弱监督跨模态特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class WeakSupervisedCrossModalEncoder(nn.Module):
    """
    弱监督跨模态编码器
    
    实现功能：
    1. 复用VLN-DUET的CrossmodalEncoder架构
    2. 属性解耦分支设计
    3. 自适应特征融合
    4. CAL对比对齐策略
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 复用VLN-DUET的核心编码器架构
        self.cross_encoder = CrossModalTransformer(config)
        
        # 属性解耦分支
        self.attribute_branches = nn.ModuleDict({
            'color': AttributeBranch(config, 'color'),
            'material': AttributeBranch(config, 'material'),
            'shape': AttributeBranch(config, 'shape')
        })
        
        # CAL对比对齐模块
        self.contrastive_aligner = ContrastiveAlignmentLearner(config)
        
        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(len(self.attribute_branches)))
        
        # 全局特征聚合器
        self.global_aggregator = GlobalFeatureAggregator(config)
        
        # 特征投影层
        self.feature_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, visual_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            visual_feats: [B, seq_len, hidden_size] 视觉特征
            text_feats: [B, text_len, hidden_size] 可选的文本特征
            
        Returns:
            fused_feats: [B, seq_len, hidden_size] 融合后特征
            attributes: Dict[str, Tensor] 属性特征字典
        """
        # 跨模态特征融合
        if text_feats is not None:
            # 有文本模态时的融合
            fused_feats = self.cross_encoder(visual_feats, text_feats)
        else:
            # 仅视觉模态时的处理
            fused_feats = self.cross_encoder(visual_feats, None)
        
        # 投影特征
        projected_feats = self.feature_projector(fused_feats)
        
        # 属性解耦
        attributes = {}
        attribute_weights = F.softmax(self.fusion_weights, dim=0)
        
        for i, (attr_name, attr_branch) in enumerate(self.attribute_branches.items()):
            attr_feat = attr_branch(projected_feats)
            # 应用自适应权重
            attr_feat = attr_feat * attribute_weights[i]
            attributes[attr_name] = attr_feat
        
        # CAL对比对齐（如果有多个属性）
        if len(attributes) > 1:
            aligned_attributes = self.contrastive_aligner(attributes, visual_feats)
            attributes.update(aligned_attributes)
        
        # 全局特征聚合
        final_fused_feats = self.global_aggregator(projected_feats, attributes)
        
        return final_fused_feats, attributes

class CrossModalTransformer(nn.Module):
    """跨模态Transformer（基于VLN-DUET架构）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 模态嵌入
        self.visual_embedding = ModalityEmbedding(config, 'visual')
        self.text_embedding = ModalityEmbedding(config, 'text')
        
        # 跨模态注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(config) 
            for _ in range(config.num_layers)
        ])
        
        # 位置编码
        self.position_encoding = PositionalEncoding(config.hidden_size)
        
        # 模态融合层
        self.modal_fusion = ModalFusionLayer(config)
        
    def forward(self, visual_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        跨模态编码
        
        Args:
            visual_feats: [B, V_len, hidden_size] 视觉特征
            text_feats: [B, T_len, hidden_size] 可选文本特征
            
        Returns:
            encoded_feats: [B, V_len, hidden_size] 编码后特征
        """
        # 视觉特征嵌入
        visual_embedded = self.visual_embedding(visual_feats)
        visual_embedded = self.position_encoding(visual_embedded)
        
        if text_feats is not None:
            # 文本特征嵌入
            text_embedded = self.text_embedding(text_feats)
            text_embedded = self.position_encoding(text_embedded)
            
            # 跨模态注意力处理
            current_visual = visual_embedded
            current_text = text_embedded
            
            for layer in self.cross_attention_layers:
                current_visual, current_text = layer(current_visual, current_text)
            
            # 模态融合
            fused_feats = self.modal_fusion(current_visual, current_text)
        else:
            # 仅视觉模态处理
            current_visual = visual_embedded
            
            for layer in self.cross_attention_layers:
                # 自注意力处理
                current_visual, _ = layer(current_visual, None)
            
            fused_feats = current_visual
        
        return fused_feats

class CrossModalAttentionLayer(nn.Module):
    """跨模态注意力层"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉自注意力
        self.visual_self_attention = MultiHeadAttention(config)
        
        # 文本自注意力
        self.text_self_attention = MultiHeadAttention(config)
        
        # 跨模态注意力（视觉->文本）
        self.visual_to_text_attention = MultiHeadAttention(config)
        
        # 跨模态注意力（文本->视觉）
        self.text_to_visual_attention = MultiHeadAttention(config)
        
        # 前馈网络
        self.visual_ffn = FeedForwardNetwork(config)
        self.text_ffn = FeedForwardNetwork(config)
        
        # 层归一化
        self.visual_norm1 = nn.LayerNorm(config.hidden_size)
        self.visual_norm2 = nn.LayerNorm(config.hidden_size)
        self.visual_norm3 = nn.LayerNorm(config.hidden_size)
        
        self.text_norm1 = nn.LayerNorm(config.hidden_size)
        self.text_norm2 = nn.LayerNorm(config.hidden_size)
        self.text_norm3 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, visual_feats: torch.Tensor, 
                text_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        跨模态注意力前向传播
        
        Args:
            visual_feats: [B, V_len, hidden_size] 视觉特征
            text_feats: [B, T_len, hidden_size] 可选文本特征
            
        Returns:
            updated_visual: [B, V_len, hidden_size] 更新后视觉特征
            updated_text: [B, T_len, hidden_size] 更新后文本特征（如果有）
        """
        # 视觉自注意力
        visual_self_attn = self.visual_self_attention(visual_feats, visual_feats, visual_feats)
        visual_feats = self.visual_norm1(visual_feats + visual_self_attn)
        
        if text_feats is not None:
            # 文本自注意力
            text_self_attn = self.text_self_attention(text_feats, text_feats, text_feats)
            text_feats = self.text_norm1(text_feats + text_self_attn)
            
            # 跨模态注意力
            # 视觉 -> 文本
            v2t_attn = self.visual_to_text_attention(visual_feats, text_feats, text_feats)
            visual_feats = self.visual_norm2(visual_feats + v2t_attn)
            
            # 文本 -> 视觉
            t2v_attn = self.text_to_visual_attention(text_feats, visual_feats, visual_feats)
            text_feats = self.text_norm2(text_feats + t2v_attn)
            
            # 前馈网络
            text_ffn_out = self.text_ffn(text_feats)
            text_feats = self.text_norm3(text_feats + text_ffn_out)
        
        # 视觉前馈网络
        visual_ffn_out = self.visual_ffn(visual_feats)
        visual_feats = self.visual_norm3(visual_feats + visual_ffn_out)
        
        return visual_feats, text_feats

class AttributeBranch(nn.Module):
    """属性分支网络"""
    
    def __init__(self, config, attribute_name: str):
        super().__init__()
        self.config = config
        self.attribute_name = attribute_name
        
        # 属性特定的特征提取器
        self.attribute_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.attr_dim),
            nn.LayerNorm(config.attr_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.attr_dim, config.attr_dim)
        )
        
        # 属性注意力机制
        self.attribute_attention = AttributeAttention(config, attribute_name)
        
        # 属性特定的归一化
        self.attribute_norm = nn.LayerNorm(config.attr_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        属性分支前向传播
        
        Args:
            features: [B, seq_len, hidden_size] 输入特征
            
        Returns:
            attribute_features: [B, seq_len, attr_dim] 属性特征
        """
        # 属性特征提取
        attr_feats = self.attribute_extractor(features)
        
        # 属性注意力
        attended_feats = self.attribute_attention(attr_feats)
        
        # 残差连接和归一化
        output = self.attribute_norm(attr_feats + attended_feats)
        
        return output

class ContrastiveAlignmentLearner(nn.Module):
    """CAL对比对齐学习器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 对比学习投影头
        self.projection_heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(config.attr_dim, config.attr_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(config.attr_dim // 2, config.attr_dim // 4)
            ) for attr in ['color', 'material', 'shape']
        })
        
        # 相关性权重预测器
        self.correlation_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size // 2, 3),  # 三个属性的相关性
            nn.Softmax(dim=-1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, attributes: Dict[str, torch.Tensor], 
                visual_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        CAL对比对齐
        
        Args:
            attributes: 属性特征字典
            visual_context: [B, seq_len, hidden_size] 视觉上下文
            
        Returns:
            aligned_attributes: 对齐后的属性特征
        """
        aligned_attributes = {}
        
        # 预测属性相关性权重
        context_pooled = visual_context.mean(dim=1)  # [B, hidden_size]
        correlation_weights = self.correlation_predictor(context_pooled)  # [B, 3]
        
        # 对每个属性进行对比对齐
        attr_names = list(attributes.keys())
        for i, (attr_name, attr_feats) in enumerate(attributes.items()):
            if attr_name in self.projection_heads:
                # 投影到对比空间
                projected = self.projection_heads[attr_name](attr_feats)
                
                # 应用相关性权重
                weight = correlation_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
                weighted_projected = projected * weight
                
                aligned_attributes[f'{attr_name}_aligned'] = weighted_projected
        
        return aligned_attributes
    
    def compute_contrastive_loss(self, attr1: torch.Tensor, attr2: torch.Tensor, 
                               labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            attr1: [B, dim] 第一个属性特征
            attr2: [B, dim] 第二个属性特征
            labels: [B] 标签（1表示正样本对，0表示负样本对）
            
        Returns:
            contrastive_loss: 对比损失
        """
        # 计算相似度
        similarity = F.cosine_similarity(attr1, attr2, dim=-1) / self.temperature
        
        # 对比损失
        pos_mask = (labels == 1).float()
        neg_mask = (labels == 0).float()
        
        pos_loss = -torch.log(torch.sigmoid(similarity) + 1e-8) * pos_mask
        neg_loss = -torch.log(torch.sigmoid(-similarity) + 1e-8) * neg_mask
        
        total_loss = (pos_loss + neg_loss).mean()
        
        return total_loss

class GlobalFeatureAggregator(nn.Module):
    """全局特征聚合器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 导入维度适配器
        from .dimension_adapter import DimensionAdapter
        
        # 动态计算总属性维度（将在运行时确定）
        self.expected_attr_dim = config.attr_dim * 3  # 预期的三个属性维度
        
        # 使用维度适配器处理可能的维度不匹配
        self.feature_adapter = DimensionAdapter(
            input_dim=config.hidden_size,  # 将在forward中动态调整
            output_dim=config.hidden_size
        )
        
        # 聚合网络（使用固定的输入维度）
        self.aggregation_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 注意力权重
        self.attention_weights = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, global_feats: torch.Tensor, 
                attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        全局特征聚合
        
        Args:
            global_feats: [B, seq_len, hidden_size] 全局特征
            attributes: 属性特征字典
            
        Returns:
            aggregated_feats: [B, seq_len, hidden_size] 聚合后特征
        """
        # 拼接属性特征
        attr_feats_list = [attr_feat.mean(dim=1) if attr_feat.dim() > 2 else attr_feat 
                          for attr_feat in attributes.values() if 'aligned' not in str(attr_feat)]
        
        if attr_feats_list:
            # 聚合属性特征
            attr_feats_concat = torch.cat(attr_feats_list, dim=-1)  # [B, total_attr_dim]
            
            # 扩展到序列维度
            attr_feats_expanded = attr_feats_concat.unsqueeze(1).expand(-1, global_feats.size(1), -1)
            
            # 拼接全局特征和属性特征
            combined_feats = torch.cat([global_feats, attr_feats_expanded], dim=-1)
            
            # 使用维度适配器处理维度不匹配
            # 动态创建适配器以匹配实际输入维度
            actual_input_dim = combined_feats.size(-1)
            if actual_input_dim != self.config.hidden_size:
                from .dimension_adapter import DimensionAdapter
                dynamic_adapter = DimensionAdapter(actual_input_dim, self.config.hidden_size)
                # 将适配器移到相同设备
                dynamic_adapter = dynamic_adapter.to(combined_feats.device)
                adapted_feats = dynamic_adapter(combined_feats)
            else:
                adapted_feats = combined_feats
            
            # 聚合
            aggregated = self.aggregation_network(adapted_feats)
            
            # 注意力权重
            attention = self.attention_weights(aggregated)
            
            # 加权融合
            final_feats = global_feats * (1 - attention) + aggregated * attention
        else:
            final_feats = global_feats
        
        return final_feats

# 辅助模块
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        Q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.output(context)
        return output

class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class ModalityEmbedding(nn.Module):
    """模态嵌入"""
    
    def __init__(self, config, modality: str):
        super().__init__()
        self.modality = modality
        self.embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(features)
        return self.layer_norm(embedded)

class ModalFusionLayer(nn.Module):
    """模态融合层"""
    
    def __init__(self, config):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, visual_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        # 池化文本特征到视觉序列长度
        text_pooled = text_feats.mean(dim=1, keepdim=True).expand(-1, visual_feats.size(1), -1)
        
        # 拼接并融合
        combined = torch.cat([visual_feats, text_pooled], dim=-1)
        fused = self.fusion(combined)
        
        return fused

class AttributeAttention(nn.Module):
    """属性注意力机制"""
    
    def __init__(self, config, attribute_name: str):
        super().__init__()
        self.attribute_name = attribute_name
        
        self.attention = nn.Sequential(
            nn.Linear(config.attr_dim, config.attr_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.attr_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重
        attention_weights = self.attention(features)  # [B, seq_len, 1]
        
        # 应用注意力
        attended = features * attention_weights
        
        return attended

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1) 