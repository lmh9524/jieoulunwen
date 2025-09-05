"""
CelebA优化配置 - 基于测试集分析结果的改进版本
"""
from dataclasses import dataclass
from typing import Dict
from .base_config import CelebAConfig

@dataclass
class CelebAOptimizedConfig(CelebAConfig):
    """CelebA优化配置 - Stage 1"""
    dataset_name: str = 'CelebA_Optimized'
    
    # 训练优化
    batch_size: int = 32  # 从16增加到32
    learning_rate: float = 8e-5  # 略微降低学习率配合更大batch
    num_epochs: int = 20  # Stage1先训练20轮
    
    # 早停与调度
    early_stopping_patience: int = 5
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    
    def __post_init__(self):
        super().__post_init__()
        
        # 重新设计的属性分组 - 基于性能分析
        self.num_classes = {
            # 拆分facial_features (10类 → 4+4类)
            'face_structure': 4,    # 基础脸型结构 (之前facial_features的子集)
            'face_details': 4,      # 面部细节特征 (之前facial_features的子集)
            
            # 简化hair_style (8类 → 5类)
            'hair_basic': 5,        # 简化的基础发型分类
            
            # 保持表现好的属性组不变
            'expression': 3,        # 87.71% - 保持
            'demographics': 4,      # 87.31% - 保持  
            'makeup': 4,           # 78.91% - 保持
            'accessories': 5,      # 86.99% - 保持
            'facial_hair': 6,      # 90.03% - 保持
            'quality': 3,          # 79.66% - 保持
        }
        
        # 基于测试结果调整权重策略
        self.loss_weights = {
            # 新拆分的属性组 - 给予更高权重
            'face_structure_cls': 1.5,   # 新分组，重点训练
            'face_details_cls': 1.3,     # 新分组，重点训练
            'hair_basic_cls': 1.4,       # 从1.0提升，需要重点改进
            
            # 表现好的属性组 - 降低权重避免过拟合
            'expression_cls': 0.8,       # 从1.0降低，已表现优秀
            'demographics_cls': 0.9,     # 从1.1降低，已表现优秀
            'makeup_cls': 0.7,          # 从0.8降低，表现良好
            'accessories_cls': 0.5,     # 从0.6降低，表现优秀
            'facial_hair_cls': 0.6,     # 从0.7降低，但需平衡召回率
            'quality_cls': 0.4,         # 从0.5降低，表现超预期
            
            # Stage 1: 保持高级模块关闭，确保稳定性
            'reg': 0.0,
            'cal': 0.0,
            'mavd': 0.0,
            'hierarchy': 0.0,
            'frequency': 0.0,
            'graph': 0.0
        }

@dataclass  
class CelebAOptimizedStage2Config(CelebAOptimizedConfig):
    """CelebA优化配置 - Stage 2 (启用轻量模块)"""
    dataset_name: str = 'CelebA_Optimized_Stage2'
    num_epochs: int = 25  # Stage2训练25轮
    
    def __post_init__(self):
        super().__post_init__()
        
        # Stage 2: 启用轻量正则模块
        self.loss_weights.update({
            'reg': 0.01,    # 启用CMDL轻量正则
            'cal': 0.005,   # 启用对比对齐
            # 其他模块继续关闭
            'mavd': 0.0,
            'hierarchy': 0.0, 
            'frequency': 0.0,
            'graph': 0.0
        })
        
        # 启用轻量正则化模块
        self.use_cmdl_regularization = True

@dataclass
class CelebAOptimizedStage3Config(CelebAOptimizedStage2Config):
    """CelebA优化配置 - Stage 3 (启用完整模块)"""
    dataset_name: str = 'CelebA_Optimized_Stage3'
    num_epochs: int = 30  # Stage3训练30轮
    
    def __post_init__(self):
        super().__post_init__()
        
        # Stage 3: 启用更多高级模块
        self.loss_weights.update({
            'reg': 0.015,     # 增加CMDL权重
            'cal': 0.008,     # 增加对比对齐权重
            'hierarchy': 0.005,  # 启用层级一致性
            'frequency': 0.003,  # 启用频域多样性
            # 保守启用
            'mavd': 0.0,     # 动态路由暂时保持关闭
            'graph': 0.0,    # 图损失暂时保持关闭
        })
        
        # 启用更多模块
        self.use_hierarchical_decomposition = True
        self.use_frequency_decoupling = True

def get_optimized_config(stage: int = 1) -> CelebAOptimizedConfig:
    """获取指定阶段的优化配置"""
    config_map = {
        1: CelebAOptimizedConfig(),
        2: CelebAOptimizedStage2Config(), 
        3: CelebAOptimizedStage3Config()
    }
    
    if stage not in config_map:
        raise ValueError(f"不支持的阶段: {stage}, 支持阶段: 1, 2, 3")
    
    return config_map[stage] 