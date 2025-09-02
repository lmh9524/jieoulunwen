"""
弱监督解耦的跨模态属性对齐 - 主要模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import torchvision.models as models

from .cross_modal_encoder import WeakSupervisedCrossModalEncoder
from .dynamic_router import MAVDDynamicRouter
from .frequency_decoupler import FrequencyDomainDecoupler
from .hierarchical_decomposer import WINNERHierarchicalDecomposer
from .regularizers import CMDLLightweightRegularizer

class WeakSupervisedCrossModalAlignment(nn.Module):
    """
    弱监督解耦的跨模态属性对齐主模型
    
    集成所有核心创新模块：
    1. MAVD动态伪标签生成
    2. CAL对比对齐策略  
    3. AFANet频域解耦
    4. WINNER层级分解
    5. CMDL轻量化正则化
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础视觉编码器
        self.visual_encoder = self._build_visual_encoder(config)
        
        # 核心创新模块
        if config.use_frequency_decoupling:
            self.freq_decoupler = FrequencyDomainDecoupler(config)
        else:
            self.freq_decoupler = None
            
        self.cross_modal_encoder = WeakSupervisedCrossModalEncoder(config)
        
        if config.use_dynamic_routing:
            self.mavd_router = MAVDDynamicRouter(config)
        else:
            self.mavd_router = None
            
        if config.use_hierarchical_decomposition:
            self.hierarchical_decomposer = WINNERHierarchicalDecomposer(config)
        else:
            self.hierarchical_decomposer = None
            
        if config.use_cmdl_regularization:
            self.cmdl_regularizer = CMDLLightweightRegularizer(config)
        else:
            self.cmdl_regularizer = None
        
        # 属性分类器
        self.attribute_classifier = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, config.num_classes[attr])
            ) for attr in config.num_classes.keys()
        })
        
        # 特征融合层
        self.feature_fusion = FeatureFusionModule(config)
        
        # 输出投影层
        self.output_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _build_visual_encoder(self, config):
        """构建视觉编码器"""
        # 使用预训练的ResNet作为视觉特征提取器
        backbone = models.resnet50(pretrained=True)
        
        # 移除最后的分类层
        modules = list(backbone.children())[:-2]  # 保留到avg_pool之前
        visual_backbone = nn.Sequential(*modules)
        
        # 添加自适应池化和投影层
        class TransposeLayer(nn.Module):
            def __init__(self, dim1, dim2):
                super().__init__()
                self.dim1 = dim1
                self.dim2 = dim2
            
            def forward(self, x):
                return x.transpose(self.dim1, self.dim2)
        
        visual_encoder = nn.Sequential(
            visual_backbone,
            nn.AdaptiveAvgPool2d((7, 7)),  # 输出 [B, 2048, 7, 7]
            nn.Flatten(2),  # [B, 2048, 49]
            TransposeLayer(1, 2),  # [B, 49, 2048]
            nn.Linear(2048, config.hidden_size),  # [B, 49, hidden_size]
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )
        
        return visual_encoder
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, images: torch.Tensor, 
                texts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 输入图像
            texts: [B, T_len, hidden_size] 可选文本特征
            
        Returns:
            outputs: 包含所有输出的字典
        """
        batch_size = images.size(0)
        outputs = {}
        
        # 1. 频域解耦（如果启用）
        if self.freq_decoupler is not None:
            freq_features = self.freq_decoupler(images)  # [B, hidden_size]
            freq_features = freq_features.unsqueeze(1)  # [B, 1, hidden_size]
            outputs['frequency_features'] = freq_features
        else:
            freq_features = None
        
        # 2. 视觉特征提取
        visual_features = self.visual_encoder(images)  # [B, seq_len, hidden_size]
        
        # 3. 特征融合（如果有频域特征）
        if freq_features is not None:
            # 扩展频域特征到序列长度
            freq_expanded = freq_features.expand(-1, visual_features.size(1), -1)
            combined_features = self.feature_fusion(visual_features, freq_expanded)
        else:
            combined_features = visual_features
        
        # 4. 跨模态编码与属性解耦
        cross_modal_features, raw_attributes = self.cross_modal_encoder(
            combined_features, texts
        )
        outputs['cross_modal_features'] = cross_modal_features
        outputs['raw_attributes'] = raw_attributes
        
        # 5. MAVD动态路由（如果启用）
        if self.mavd_router is not None:
            # 池化特征用于路由
            pooled_features = cross_modal_features.mean(dim=1)  # [B, hidden_size]
            pseudo_labels, importance_weights = self.mavd_router(pooled_features)
            outputs['pseudo_labels'] = pseudo_labels
            outputs['importance_weights'] = importance_weights
        
        # 6. WINNER层级分解（如果启用）
        if self.hierarchical_decomposer is not None:
            hierarchical_features, graph_representation = self.hierarchical_decomposer(
                cross_modal_features
            )
            outputs['hierarchical_features'] = hierarchical_features
            outputs['graph_representation'] = graph_representation
            
            # 使用层级特征进行最终预测
            final_features = hierarchical_features[-1]  # 使用最高层级特征
        else:
            final_features = cross_modal_features
        
        # 7. 输出投影
        projected_features = self.output_projector(final_features)
        
        # 8. 属性预测
        predictions = {}
        pooled_projected = projected_features.mean(dim=1)  # [B, hidden_size]
        
        for attr_name, classifier in self.attribute_classifier.items():
            attr_logits = classifier(pooled_projected)  # [B, num_classes]
            predictions[attr_name] = attr_logits
        
        outputs['predictions'] = predictions
        outputs['final_features'] = projected_features
        
        # 9. CMDL正则化损失（如果启用）
        if self.cmdl_regularizer is not None and raw_attributes:
            # 准备属性特征用于正则化
            regularization_attributes = {}
            for attr_name, attr_feat in raw_attributes.items():
                if 'aligned' not in attr_name:  # 排除对齐后的特征
                    # 池化到固定维度
                    if attr_feat.dim() > 2:
                        pooled_attr = attr_feat.mean(dim=1)  # [B, attr_dim]
                    else:
                        pooled_attr = attr_feat
                    regularization_attributes[attr_name] = pooled_attr
            
            if regularization_attributes:
                regularization_loss = self.cmdl_regularizer(regularization_attributes)
                outputs['regularization_loss'] = regularization_loss
        
        return outputs
    
    def compute_total_loss(self, outputs: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            
        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0
        loss_components = {}
        
        # 1. 分类损失
        if 'predictions' in outputs:
            classification_loss = 0.0
            for attr_name, pred_logits in outputs['predictions'].items():
                if attr_name in targets:
                    attr_loss = F.cross_entropy(pred_logits, targets[attr_name])
                    classification_loss += self.config.loss_weights.get(f'{attr_name}_cls', 1.0) * attr_loss
                    loss_components[f'{attr_name}_loss'] = attr_loss
            
            total_loss += classification_loss
            loss_components['classification_loss'] = classification_loss
        
        # 2. 正则化损失
        if 'regularization_loss' in outputs:
            reg_loss = outputs['regularization_loss']
            total_loss += self.config.loss_weights.get('reg', 0.1) * reg_loss
            loss_components['regularization_loss'] = reg_loss
        
        # 3. 对比对齐损失（如果有伪标签）
        if 'pseudo_labels' in outputs and 'alignment_targets' in targets:
            cal_loss = self._compute_contrastive_alignment_loss(
                outputs['pseudo_labels'],
                targets['alignment_targets'],
                targets.get('alignment_labels', None)
            )
            total_loss += self.config.loss_weights.get('cal', 0.05) * cal_loss
            loss_components['contrastive_alignment_loss'] = cal_loss
        
        # 4. 层级一致性损失
        if 'hierarchical_features' in outputs:
            hierarchy_loss = self._compute_hierarchy_consistency_loss(
                outputs['hierarchical_features']
            )
            total_loss += 0.01 * hierarchy_loss
            loss_components['hierarchy_consistency_loss'] = hierarchy_loss
        
        # 5. 频域解耦损失
        if 'frequency_features' in outputs:
            freq_loss = self._compute_frequency_diversity_loss(
                outputs['frequency_features']
            )
            total_loss += 0.005 * freq_loss
            loss_components['frequency_diversity_loss'] = freq_loss
        
        loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components
    
    def _compute_contrastive_alignment_loss(self, pseudo_labels: torch.Tensor,
                                          alignment_targets: torch.Tensor,
                                          alignment_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算对比对齐损失"""
        if alignment_labels is None:
            # 如果没有提供标签，使用伪标签生成
            alignment_labels = (torch.sum(pseudo_labels * alignment_targets, dim=-1) > 0.5).float()
        
        # 计算相似度
        similarity = F.cosine_similarity(
            pseudo_labels.unsqueeze(1), 
            alignment_targets.unsqueeze(0), 
            dim=-1
        )
        
        # 对比损失
        pos_mask = alignment_labels.unsqueeze(1)
        neg_mask = 1 - pos_mask
        
        pos_loss = -torch.log(torch.sigmoid(similarity) + 1e-8) * pos_mask
        neg_loss = -torch.log(torch.sigmoid(-similarity) + 1e-8) * neg_mask
        
        return (pos_loss + neg_loss).mean()
    
    def _compute_hierarchy_consistency_loss(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """计算层级一致性损失"""
        consistency_loss = 0.0
        
        for i in range(len(hierarchical_features) - 1):
            current_level = hierarchical_features[i].mean(dim=1)
            next_level = hierarchical_features[i + 1].mean(dim=1)
            
            # 层级间的平滑性约束
            level_diff = F.mse_loss(current_level, next_level)
            consistency_loss += level_diff
        
        return consistency_loss / max(1, len(hierarchical_features) - 1)
    
    def _compute_frequency_diversity_loss(self, frequency_features: torch.Tensor) -> torch.Tensor:
        """计算频域多样性损失"""
        # 鼓励频域特征的多样性
        batch_size = frequency_features.size(0)
        
        # 计算特征间的相似度矩阵
        normalized_features = F.normalize(frequency_features.squeeze(1), p=2, dim=-1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # 多样性损失：减少非对角线元素
        mask = torch.eye(batch_size, device=frequency_features.device)
        diversity_loss = torch.sum(similarity_matrix * (1 - mask)) / (batch_size * (batch_size - 1))
        
        return diversity_loss
    
    def get_attention_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取注意力图（用于可视化）
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            attention_maps: 注意力图字典
        """
        self.eval()
        attention_maps = {}
        
        with torch.no_grad():
            # 获取视觉特征
            visual_features = self.visual_encoder(images)
            
            # 跨模态编码器的注意力
            cross_modal_features, raw_attributes = self.cross_modal_encoder(visual_features)
            
            # 属性特定的注意力图
            for attr_name, attr_feat in raw_attributes.items():
                if 'aligned' not in attr_name and attr_feat.dim() == 3:
                    # 计算注意力权重
                    attention_weights = torch.mean(attr_feat**2, dim=-1)  # [B, seq_len]
                    attention_weights = F.softmax(attention_weights, dim=-1)
                    attention_maps[f'{attr_name}_attention'] = attention_weights
            
            # 如果有层级分解，获取层级注意力
            if self.hierarchical_decomposer is not None:
                hierarchical_features, _ = self.hierarchical_decomposer(cross_modal_features)
                for i, level_feat in enumerate(hierarchical_features):
                    level_attention = torch.mean(level_feat**2, dim=-1)
                    level_attention = F.softmax(level_attention, dim=-1)
                    attention_maps[f'level_{i}_attention'] = level_attention
        
        return attention_maps
    
    def extract_features(self, images: torch.Tensor, 
                        layer_name: str = 'final') -> torch.Tensor:
        """
        提取指定层的特征（用于特征分析）
        
        Args:
            images: [B, C, H, W] 输入图像
            layer_name: 要提取的层名称
            
        Returns:
            features: 提取的特征
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(images)
            
            if layer_name == 'visual':
                return self.visual_encoder(images)
            elif layer_name == 'cross_modal':
                return outputs['cross_modal_features']
            elif layer_name == 'final':
                return outputs['final_features']
            elif layer_name in outputs:
                return outputs[layer_name]
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")

class FeatureFusionModule(nn.Module):
    """特征融合模块"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
        """
        特征融合
        
        Args:
            feature1: [B, seq_len, hidden_size] 第一个特征
            feature2: [B, seq_len, hidden_size] 第二个特征
            
        Returns:
            fused_features: [B, seq_len, hidden_size] 融合后特征
        """
        # 拼接特征
        concatenated = torch.cat([feature1, feature2], dim=-1)
        
        # 融合
        fused = self.fusion_network(concatenated)
        
        # 门控
        gate_weights = self.gate(concatenated)
        
        # 加权融合
        output = gate_weights * fused + (1 - gate_weights) * feature1
        
        return output 