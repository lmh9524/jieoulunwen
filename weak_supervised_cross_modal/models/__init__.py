"""
弱监督解耦的跨模态属性对齐 - 模型模块
"""

from .base_model import WeakSupervisedCrossModalAlignment
from .cross_modal_encoder import WeakSupervisedCrossModalEncoder
from .dynamic_router import MAVDDynamicRouter
from .frequency_decoupler import FrequencyDomainDecoupler
from .hierarchical_decomposer import WINNERHierarchicalDecomposer
from .regularizers import CMDLLightweightRegularizer
from .dimension_adapter import DimensionAdapter, MultiDimensionAdapter, AdaptiveFeatureFusion

__all__ = [
    'WeakSupervisedCrossModalAlignment',
    'WeakSupervisedCrossModalEncoder', 
    'MAVDDynamicRouter',
    'FrequencyDomainDecoupler',
    'WINNERHierarchicalDecomposer',
    'CMDLLightweightRegularizer',
    'DimensionAdapter',
    'MultiDimensionAdapter',
    'AdaptiveFeatureFusion'
] 