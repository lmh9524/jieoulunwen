"""
弱监督解耦的跨模态属性对齐 - 基础配置文件
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BaseConfig:
    """基础配置类"""
    
    # ===== 模型架构参数 =====
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 6
    intermediate_size: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # ===== 属性相关参数 =====
    attr_dim: int = 256
    num_experts: int = 5  # MAVD动态路由专家数量
    num_levels: int = 3   # WINNER层级分解层数
    graph_dim: int = 512  # 属性关系图维度
    
    # ===== 频域解耦参数 =====
    freq_cutoff: float = 0.3  # AFANet频域分解截止频率
    high_freq_dim: int = 384
    low_freq_dim: int = 384
    
    # ===== 训练参数 =====
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # ===== 损失权重 =====
    loss_weights: Dict[str, float] = None
    
    # ===== CMDL轻量化正则化参数 =====
    cmdl_lambda: float = 0.1
    mi_hidden_dim: int = 512
    
    # ===== 数据集参数 =====
    dataset_name: str = 'CUB'
    data_path: str = './data'
    image_size: int = 224
    num_classes: Dict[str, int] = None

    def __post_init__(self):
        """初始化后处理 - 合并所有初始化逻辑"""
        # 初始化损失权重
        if self.loss_weights is None:
            self.loss_weights = {
                'color_cls': 1.0,
                'material_cls': 1.0, 
                'shape_cls': 1.0,
                'reg': 0.1,          # CMDL正则化权重
                'cal': 0.05,         # CAL对比对齐权重
                'mavd': 0.03         # MAVD动态路由权重
            }
        
        # 初始化类别数量
        if self.num_classes is None:
            self.num_classes = {
                'color': 11,      # CUB数据集颜色类别数
                'material': 8,    # 材质类别数
                'shape': 15       # 形状类别数
            }
    
    # ===== 设备配置 =====
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    
    # ===== 实验配置 =====
    experiment_name: str = 'weak_supervised_cross_modal'
    save_dir: str = './experiments/results'
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # ===== 模型特定参数 =====
    use_frequency_decoupling: bool = True
    use_hierarchical_decomposition: bool = True
    use_dynamic_routing: bool = True
    use_cmdl_regularization: bool = True
    
    # ===== 可视化参数 =====
    visualize_attention: bool = True
    visualize_features: bool = True
    save_visualizations: bool = True

@dataclass 
class CUBConfig(BaseConfig):
    """CUB数据集特定配置"""
    dataset_name: str = 'CUB'
    image_size: int = 224
    num_classes: Dict[str, int] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.num_classes = {
            'color': 11,
            'material': 8, 
            'shape': 15
        }

@dataclass
class COCOAttributesConfig(BaseConfig):
    """COCOAttributes数据集特定配置"""
    dataset_name: str = 'COCOAttributes'
    data_path: str = '../data'  # 修改为相对于当前目录的正确路径
    image_size: int = 224
    num_classes: Dict[str, int] = None

    # COCOAttributes特定参数
    num_attributes: int = 204  # COCOAttributes有204个二进制属性
    batch_size: int = 32  # 适合COCOAttributes的批次大小
    learning_rate: float = 1e-4  # 适合多属性学习的学习率

    def __post_init__(self):
        super().__post_init__()
        # COCOAttributes的属性分组
        self.num_classes = {
            'color': 12,      # 颜色属性类别
            'material': 15,   # 材质属性类别
            'shape': 20,      # 形状属性类别
            'texture': 10,    # 纹理属性类别
            'size': 5,        # 大小属性类别
            'other': 8        # 其他属性类别
        }

@dataclass
class CelebAConfig(BaseConfig):
    """CelebA数据集特定配置"""
    dataset_name: str = 'CelebA'
    data_path: str = '/autodl-pub/data/CelebA'  # CelebA数据集路径
    image_size: int = 224
    num_classes: Dict[str, int] = None
    
    # CelebA特定参数
    num_attributes: int = 40  # CelebA有40个面部属性
    batch_size: int = 32  # 适合CelebA的批次大小
    learning_rate: float = 1e-4  # 适合人脸属性学习的学习率
    
    def __post_init__(self):
        super().__post_init__()
        # CelebA的属性分组 - 对应数据集适配器中的分类
        self.num_classes = {
            'hair_style': 8,      # 头发相关属性
            'facial_features': 10,  # 面部特征
            'makeup': 4,          # 化妆相关
            'accessories': 5,     # 配饰
            'expression': 3,      # 表情相关
            'demographics': 4,    # 性别和年龄
            'facial_hair': 6,     # 胡须相关
            'quality': 3          # 吸引力和图像质量
        }
        
        # 关闭复杂正则与路由以提升稳定性（CelebA首训仅做分类）
        self.use_frequency_decoupling = False
        self.use_hierarchical_decomposition = False
        self.use_dynamic_routing = False
        self.use_cmdl_regularization = False
        
        # 调整损失权重以适应CelebA（仅保留分类）
        self.loss_weights = {
            'hair_style_cls': 1.0,
            'facial_features_cls': 1.2,  # 面部特征更重要
            'makeup_cls': 0.8,
            'accessories_cls': 0.6,
            'expression_cls': 1.0,
            'demographics_cls': 1.1,     # 性别年龄识别重要
            'facial_hair_cls': 0.7,
            'quality_cls': 0.5,
            'reg': 0.0,                   # 关闭CMDL正则
            'cal': 0.0,                   # 关闭CAL
            'mavd': 0.0,                  # 关闭MAVD
            'hierarchy': 0.0,             # 关闭层级一致性
            'frequency': 0.0,             # 关闭频域损失
            'graph': 0.0                  # 关闭图损失
        }

def get_config(dataset_name: str = 'CUB') -> BaseConfig:
    """根据数据集名称获取对应配置"""
    config_map = {
        'CUB': CUBConfig(),
        'COCOAttributes': COCOAttributesConfig(),
        'COCO-Attributes': COCOAttributesConfig(),  # 兼容性别名
        'CelebA': CelebAConfig()
    }

    if dataset_name not in config_map:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return config_map[dataset_name]