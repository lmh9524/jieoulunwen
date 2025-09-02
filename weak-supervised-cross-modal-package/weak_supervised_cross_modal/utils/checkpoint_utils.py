"""
模型检查点工具
"""
import torch
import os
from typing import Dict, Any, Optional
import logging

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   save_path: str) -> None:
    """
    保存模型检查点
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前训练轮次
        loss: 当前损失值
        metrics: 评估指标字典
        save_path: 保存路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'model_config': getattr(model, 'config', None)  # 保存模型配置（如果存在）
    }
    
    # 如果提供了调度器，保存其状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存检查点
    try:
        torch.save(checkpoint, save_path)
        logger = logging.getLogger(__name__)
        logger.info(f"检查点已保存到: {save_path}")
        logger.info(f"  轮次: {epoch}")
        logger.info(f"  损失: {loss:.4f}")
        if metrics:
            logger.info(f"  指标: {metrics}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"保存检查点失败: {e}")
        raise

def load_checkpoint(checkpoint_path: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: PyTorch模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        
    Returns:
        checkpoint: 检查点字典
    """
    logger = logging.getLogger(__name__)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    try:
        # 加载检查点（使用CPU以确保兼容性）
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型状态已从检查点加载: {checkpoint_path}")
        
        # 如果提供了优化器且检查点中包含优化器状态，则加载
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("优化器状态已加载")
        
        # 如果提供了调度器且检查点中包含调度器状态，则加载
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("学习率调度器状态已加载")
        
        # 记录加载信息
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"检查点加载完成:")
        logger.info(f"  轮次: {epoch}")
        logger.info(f"  损失: {loss:.4f}")
        if metrics:
            logger.info(f"  指标: {metrics}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        raise

def save_best_checkpoint(model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                        epoch: int,
                        loss: float,
                        metrics: Dict[str, float],
                        save_dir: str,
                        metric_name: str = 'mean_accuracy',
                        is_better_fn=lambda x, y: x > y) -> str:
    """
    保存最佳检查点（基于指定指标）
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前训练轮次
        loss: 当前损失值
        metrics: 评估指标字典
        save_dir: 保存目录
        metric_name: 用于判断最佳模型的指标名称
        is_better_fn: 判断是否更好的函数
        
    Returns:
        save_path: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 最佳检查点路径
    best_checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    
    # 检查是否需要保存（基于指标）
    should_save = True
    if os.path.exists(best_checkpoint_path):
        try:
            best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            best_metrics = best_checkpoint.get('metrics', {})
            
            if metric_name in best_metrics and metric_name in metrics:
                current_metric = metrics[metric_name]
                best_metric = best_metrics[metric_name]
                should_save = is_better_fn(current_metric, best_metric)
        except Exception as e:
            logging.getLogger(__name__).warning(f"无法读取现有最佳检查点: {e}")
    
    if should_save:
        save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, best_checkpoint_path)
        logging.getLogger(__name__).info(f"新的最佳模型已保存 ({metric_name}: {metrics.get(metric_name, 'N/A')})")
    
    return best_checkpoint_path

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新的检查点文件路径
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        latest_checkpoint_path: 最新检查点路径，如果没有则返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，获取最新的
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def cleanup_old_checkpoints(checkpoint_dir: str, keep_count: int = 5):
    """
    清理旧的检查点文件，只保留最新的几个
    
    Args:
        checkpoint_dir: 检查点目录
        keep_count: 保留的检查点数量
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pth') and f != 'best_model.pth']
    
    if len(checkpoint_files) <= keep_count:
        return
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # 删除旧的检查点
    files_to_delete = checkpoint_files[keep_count:]
    for file_name in files_to_delete:
        file_path = os.path.join(checkpoint_dir, file_name)
        try:
            os.remove(file_path)
            logging.getLogger(__name__).info(f"已删除旧检查点: {file_name}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"删除检查点失败 {file_name}: {e}") 