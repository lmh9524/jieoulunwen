"""
配置加载器
"""
import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path is None:
            # 返回默认配置
            return self._get_default_config()
        
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data': {
                'root_path': './data/CUB_200_2011',
                'batch_size': 32,
                'num_workers': 4,
                'image_size': 224,
                'use_augmentation': True
            },
            'model': {
                'visual_dim': 2048,
                'text_dim': 312,
                'hidden_dim': 512,
                'output_dim': 256,
                'num_classes': 200,
                'num_attributes': 312,
                'dropout': 0.1,
                'use_frequency_decoupling': False,
                'use_dynamic_routing': False,
                'use_hierarchical_decomposition': False,
                'use_cmdl_regularization': False
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'step_size': 10,
                'gamma': 0.1,
                'save_interval': 10
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'save_predictions': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """支持in操作"""
        return key in self.config
    
    def keys(self):
        """返回所有键"""
        return self.config.keys()
    
    def items(self):
        """返回所有键值对"""
        return self.config.items()
    
    def values(self):
        """返回所有值"""
        return self.config.values()
    
    def update(self, other: Dict[str, Any]):
        """更新配置"""
        self.config.update(other)
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True) 