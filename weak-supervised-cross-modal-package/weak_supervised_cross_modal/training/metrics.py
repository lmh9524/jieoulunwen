"""
评估指标模块 - 属性预测性能评估
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from typing import Dict, List, Tuple, Optional

class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self, num_classes: Dict[str, int]):
        """
        初始化评估指标计算器
        
        Args:
            num_classes: 每个属性的类别数量字典
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有内部存储的预测结果和目标标签列表"""
        self.predictions = {attr: [] for attr in self.num_classes.keys()}
        self.targets = {attr: [] for attr in self.num_classes.keys()}
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]):
        """
        更新预测结果
        
        Args:
            predictions: 模型输出的预测logits字典
            targets: 真实目标标签字典
        """
        for attr_name in self.num_classes.keys():
            if attr_name in predictions and attr_name in targets:
                # 将logits转换为类别ID
                pred = torch.argmax(predictions[attr_name], dim=-1)
                self.predictions[attr_name].extend(pred.cpu().numpy())
                self.targets[attr_name].extend(targets[attr_name].cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Returns:
            metrics: 包含各项指标的字典
        """
        metrics = {}
        
        for attr_name in self.num_classes.keys():
            if len(self.predictions[attr_name]) > 0:
                pred = np.array(self.predictions[attr_name])
                target = np.array(self.targets[attr_name])
                
                # 准确率
                acc = accuracy_score(target, pred)
                metrics[f'{attr_name}_accuracy'] = acc
                
                # 精确率、召回率、F1分数
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target, pred, average='macro', zero_division=0
                )
                metrics[f'{attr_name}_precision'] = precision
                metrics[f'{attr_name}_recall'] = recall
                metrics[f'{attr_name}_f1'] = f1
        
        # 计算平均指标
        attr_accuracies = [metrics[f'{attr}_accuracy'] for attr in self.num_classes.keys() 
                          if f'{attr}_accuracy' in metrics]
        if attr_accuracies:
            metrics['mean_accuracy'] = np.mean(attr_accuracies)
        
        return metrics
    
    def get_summary(self) -> str:
        """
        获取指标摘要字符串
        
        Returns:
            summary: 格式化的指标摘要
        """
        metrics = self.compute()
        summary_lines = []
        
        for attr_name in self.num_classes.keys():
            if f'{attr_name}_accuracy' in metrics:
                acc = metrics[f'{attr_name}_accuracy']
                f1 = metrics[f'{attr_name}_f1']
                summary_lines.append(f"{attr_name}: Acc={acc:.4f}, F1={f1:.4f}")
        
        if 'mean_accuracy' in metrics:
            summary_lines.append(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        
        return "\n".join(summary_lines) 