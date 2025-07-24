"""
弱监督解耦的跨模态属性对齐 - 训练模块
"""

from .losses import ComprehensiveLoss
from .metrics import EvaluationMetrics

__all__ = [
    'ComprehensiveLoss', 
    'EvaluationMetrics'
] 