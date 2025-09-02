"""
VAW数据集专用配置文件
"""

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class VAWConfig:
    """VAW数据集训练配置"""
    
    # 数据集参数
    dataset_name: str = "VAW"
    vaw_annotations_dir: str = "D:/KKK/data/VAW/annotations"
    vaw_images_dir: str = "D:/KKK/data/VAW/images"
    
    # 模型参数
    attr_dim: int = 620
    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.15
    
    # 训练参数
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-5
    
    # 设备参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 实验参数
    experiment_name: str = "vaw_baseline"
    seed: int = 42
    
    # 调试参数
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

def get_vaw_fast_config() -> VAWConfig:
    """获取快速测试配置"""
    return VAWConfig(
        experiment_name="vaw_fast",
        batch_size=4,
        num_epochs=2,
        max_train_samples=20,
        max_val_samples=10,
        max_test_samples=10
    )
