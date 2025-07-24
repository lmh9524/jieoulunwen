"""
维度适配器模块 - 确保不同模块间的维度一致性
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import logging

class DimensionAdapter(nn.Module):
    """
    维度适配器 - 自动处理不同模块间的维度转换
    
    该类用于解决模型中不同组件间特征维度不匹配的问题，
    特别是在频域解耦、层级分解等模块之间的特征传递。
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 use_bias: bool = True,
                 activation: Optional[str] = None,
                 dropout: float = 0.0):
        """
        初始化维度适配器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            use_bias: 是否使用偏置项
            activation: 激活函数类型 ('relu', 'gelu', 'tanh', None)
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # 如果输入和输出维度相同，使用恒等映射
        if input_dim == output_dim:
            self.adapter = nn.Identity()
            self.is_identity = True
        else:
            # 创建线性层进行维度转换
            layers = []
            
            # 主要的线性变换层
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            
            # 可选的激活函数
            if activation is not None:
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation.lower() == 'gelu':
                    layers.append(nn.GELU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    logging.warning(f"未知的激活函数: {activation}，将忽略")
            
            # 可选的Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            self.adapter = nn.Sequential(*layers)
            self.is_identity = False
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        if not self.is_identity:
            for module in self.adapter.modules():
                if isinstance(module, nn.Linear):
                    # 使用Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, ..., input_dim] 或 [batch_size, seq_len, input_dim]
            
        Returns:
            output: 输出张量，形状为 [batch_size, ..., output_dim] 或 [batch_size, seq_len, output_dim]
        """
        # 验证输入维度
        if x.size(-1) != self.input_dim:
            raise ValueError(f"输入张量的最后一维应为 {self.input_dim}，但得到 {x.size(-1)}")
        
        # 应用维度适配
        output = self.adapter(x)
        
        return output
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        Returns:
            output_dim: 输出特征维度
        """
        return self.output_dim
    
    def get_input_dim(self) -> int:
        """
        获取输入维度
        
        Returns:
            input_dim: 输入特征维度
        """
        return self.input_dim
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, is_identity={self.is_identity}'

class MultiDimensionAdapter(nn.Module):
    """
    多维度适配器 - 处理多个不同维度的输入
    
    用于同时适配多个不同维度的特征，例如来自不同模块的特征融合。
    """
    
    def __init__(self, input_dims: list, output_dim: int, 
                 fusion_method: str = 'concat',
                 use_attention: bool = False):
        """
        初始化多维度适配器
        
        Args:
            input_dims: 输入维度列表
            output_dim: 输出维度
            fusion_method: 融合方法 ('concat', 'add', 'weighted_add')
            use_attention: 是否使用注意力机制进行融合
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        self.use_attention = use_attention
        
        # 为每个输入创建单独的适配器
        self.adapters = nn.ModuleList([
            DimensionAdapter(dim, output_dim) for dim in input_dims
        ])
        
        # 根据融合方法设置最终的输出层
        if fusion_method == 'concat':
            # 拼接后需要额外的线性层
            concat_dim = output_dim * len(input_dims)
            self.output_layer = nn.Linear(concat_dim, output_dim)
        elif fusion_method in ['add', 'weighted_add']:
            # 直接相加，不需要额外的线性层
            self.output_layer = nn.Identity()
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        # 可选的注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True
            )
        
        # 权重参数（用于weighted_add）
        if fusion_method == 'weighted_add':
            self.fusion_weights = nn.Parameter(torch.ones(len(input_dims)))
    
    def forward(self, inputs: list) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入张量列表，每个张量的最后一维应对应input_dims
            
        Returns:
            output: 融合后的输出张量
        """
        if len(inputs) != len(self.input_dims):
            raise ValueError(f"输入数量 ({len(inputs)}) 与预期数量 ({len(self.input_dims)}) 不匹配")
        
        # 分别适配每个输入
        adapted_inputs = []
        for i, (inp, adapter) in enumerate(zip(inputs, self.adapters)):
            adapted = adapter(inp)
            adapted_inputs.append(adapted)
        
        # 根据融合方法合并特征
        if self.fusion_method == 'concat':
            # 在最后一维拼接
            fused = torch.cat(adapted_inputs, dim=-1)
            output = self.output_layer(fused)
        
        elif self.fusion_method == 'add':
            # 直接相加
            output = torch.stack(adapted_inputs, dim=0).sum(dim=0)
        
        elif self.fusion_method == 'weighted_add':
            # 加权相加
            weights = torch.softmax(self.fusion_weights, dim=0)
            weighted_inputs = [w * inp for w, inp in zip(weights, adapted_inputs)]
            output = torch.stack(weighted_inputs, dim=0).sum(dim=0)
        
        # 可选的注意力机制
        if self.use_attention:
            # 将所有适配后的输入作为序列进行注意力计算
            stacked_inputs = torch.stack(adapted_inputs, dim=-2)  # [B, ..., num_inputs, dim]
            
            # 重塑为注意力所需的格式
            original_shape = stacked_inputs.shape[:-2]
            batch_size = stacked_inputs.size(0)
            seq_len = stacked_inputs.size(-2)
            embed_dim = stacked_inputs.size(-1)
            
            # 展平除了最后两维的所有维度
            flattened = stacked_inputs.view(-1, seq_len, embed_dim)
            
            # 应用注意力
            attended, _ = self.attention(flattened, flattened, flattened)
            
            # 恢复原始形状并取平均
            attended = attended.view(*original_shape, seq_len, embed_dim)
            output = attended.mean(dim=-2)
        
        return output

class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    
    根据输入特征的统计信息自动调整融合策略。
    """
    
    def __init__(self, input_dims: list, output_dim: int):
        """
        初始化自适应特征融合模块
        
        Args:
            input_dims: 输入维度列表
            output_dim: 输出维度
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # 维度适配器
        self.adapters = nn.ModuleList([
            DimensionAdapter(dim, output_dim) for dim in input_dims
        ])
        
        # 自适应权重网络
        # 每个输入会产生均值和标准差，所以维度翻倍
        total_global_dim = sum(input_dims) * 2
        self.weight_network = nn.Sequential(
            nn.Linear(total_global_dim, total_global_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_global_dim // 4, len(input_dims)),
            nn.Softmax(dim=-1)
        )
        
        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, inputs: list) -> torch.Tensor:
        """
        自适应特征融合前向传播
        
        Args:
            inputs: 输入特征列表
            
        Returns:
            output: 融合后的特征
        """
        # 计算全局特征用于权重预测
        global_features = []
        for inp in inputs:
            # 对每个输入计算全局统计量
            global_feat = torch.cat([
                inp.mean(dim=tuple(range(1, inp.dim()-1)), keepdim=True).squeeze(),
                inp.std(dim=tuple(range(1, inp.dim()-1)), keepdim=True).squeeze()
            ], dim=-1)
            global_features.append(global_feat)
        
        # 拼接所有全局特征
        concat_global = torch.cat(global_features, dim=-1)
        
        # 预测融合权重
        fusion_weights = self.weight_network(concat_global)  # [B, num_inputs]
        
        # 适配各个输入
        adapted_inputs = []
        for inp, adapter in zip(inputs, self.adapters):
            adapted = adapter(inp)
            adapted_inputs.append(adapted)
        
        # 加权融合
        weighted_sum = torch.zeros_like(adapted_inputs[0])
        for i, adapted in enumerate(adapted_inputs):
            weight = fusion_weights[:, i:i+1]
            # 扩展权重维度以匹配特征张量
            for _ in range(adapted.dim() - weight.dim()):
                weight = weight.unsqueeze(-1)
            weighted_sum += weight * adapted
        
        # 最终输出处理
        output = self.output_layer(weighted_sum)
        
        return output 