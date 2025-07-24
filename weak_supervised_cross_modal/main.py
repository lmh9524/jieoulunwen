"""
弱监督解耦的跨模态属性对齐 - 主训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import os
from typing import Dict, Any
import numpy as np

# 导入项目模块
from config.base_config import get_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.dataset_adapters import DatasetAdapter, CUBDatasetAdapter, COCOAttributesDatasetAdapter
from utils.logging_utils import setup_logging
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

def setup_device(device_name: str = 'auto') -> torch.device:
    """
    设置计算设备
    
    Args:
        device_name: 设备名称，'auto'表示自动选择
        
    Returns:
        device: torch设备对象
    """
    logger = logging.getLogger(__name__)
    
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.info("使用CPU")
    
    return device

def create_dataloaders(config, args) -> Dict[str, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        args: 命令行参数
        
    Returns:
        dataloaders: 数据加载器字典
    """
    if config.dataset_name == 'CUB':
        adapter = CUBDatasetAdapter(config)
    elif config.dataset_name == 'COCO-Attributes':
        adapter = COCOAttributesDatasetAdapter(config)
    else:
        raise ValueError(f"不支持的数据集: {config.dataset_name}")
    
    return adapter.get_dataloaders()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='弱监督解耦的跨模态属性对齐训练')
    
    # 基础配置
    parser.add_argument('--dataset', type=str, default='CUB', 
                       choices=['CUB', 'COCO-Attributes'],
                       help='数据集名称')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='./experiments/results',
                       help='模型保存路径')
    parser.add_argument('--experiment_name', type=str, default='weak_cross_modal',
                       help='实验名称')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    
    # 模型配置
    parser.add_argument('--use_frequency_decoupling', action='store_true',
                       help='是否使用频域解耦')
    parser.add_argument('--use_hierarchical_decomposition', action='store_true',
                       help='是否使用层级分解')
    parser.add_argument('--use_dynamic_routing', action='store_true',
                       help='是否使用动态路由')
    parser.add_argument('--use_cmdl_regularization', action='store_true',
                       help='是否使用CMDL正则化')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 其他配置
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='模型保存间隔')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """设置训练设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device

def create_dataloaders(config, args) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    dataset_adapter = DatasetAdapter(args.dataset, args.data_path)
    
    dataloaders = {}
    
    # 训练集
    train_dataset = dataset_adapter.create_dataloader(
        batch_size=args.batch_size,
        split='train',
        num_workers=args.num_workers
    )
    dataloaders['train'] = train_dataset
    
    # 验证集
    val_dataset = dataset_adapter.create_dataloader(
        batch_size=args.batch_size,
        split='val',
        num_workers=args.num_workers
    )
    dataloaders['val'] = val_dataset
    
    # 测试集
    test_dataset = dataset_adapter.create_dataloader(
        batch_size=args.batch_size,
        split='test',
        num_workers=args.num_workers
    )
    dataloaders['test'] = test_dataset
    
    return dataloaders

def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                args) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    loss_components_sum = {}
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据移动到设备
        images = batch['images'].to(device)
        targets = {k: v.to(device) if torch.is_tensor(v) else v 
                  for k, v in batch['targets'].items()}
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        loss, loss_components = criterion(outputs, targets, epoch)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 优化器步骤
        optimizer.step()
        
        # 统计
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 累积损失组件
        for key, value in loss_components.items():
            if key not in loss_components_sum:
                loss_components_sum[key] = 0.0
            if torch.is_tensor(value):
                loss_components_sum[key] += value.item() * batch_size
            else:
                loss_components_sum[key] += value * batch_size
        
        # 打印日志
        if batch_idx % args.log_interval == 0:
            print(f'训练 Epoch: {epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                  f'损失: {loss.item():.6f}')
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    avg_loss_components = {k: v / total_samples for k, v in loss_components_sum.items()}
    
    return {'avg_loss': avg_loss, **avg_loss_components}

def validate_epoch(model: nn.Module,
                  dataloader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  epoch: int) -> Dict[str, float]:
    """验证一个epoch"""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    metrics = EvaluationMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            # 数据移动到设备
            images = batch['images'].to(device)
            targets = {k: v.to(device) if torch.is_tensor(v) else v 
                      for k, v in batch['targets'].items()}
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss, _ = criterion(outputs, targets, epoch)
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 更新评估指标
            metrics.update(outputs, targets)
    
    # 计算指标
    avg_loss = total_loss / total_samples
    metric_results = metrics.compute_metrics()
    
    return {'val_loss': avg_loss, **metric_results}

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.save_dir, args.experiment_name)
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = setup_device('auto')
    
    # 获取配置
    config = get_config(args.dataset)
    
    # 更新配置
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_epochs = args.num_epochs
    config.use_frequency_decoupling = args.use_frequency_decoupling
    config.use_hierarchical_decomposition = args.use_hierarchical_decomposition
    config.use_dynamic_routing = args.use_dynamic_routing
    config.use_cmdl_regularization = args.use_cmdl_regularization
    config.device = device
    
    logger.info(f"配置: {config}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    dataloaders = create_dataloaders(config, args)
    
    # 创建模型
    logger.info("创建模型...")
    model = WeakSupervisedCrossModalAlignment(config)
    model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数
    criterion = ComprehensiveLoss(config)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs
    )
    
    # 恢复训练（如果需要）
    start_epoch = 0
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 仅评估模式
    if args.eval_only:
        logger.info("开始评估...")
        val_results = validate_epoch(model, dataloaders['test'], criterion, device, 0)
        logger.info(f"评估结果: {val_results}")
        return
    
    # 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        
        # 训练
        train_results = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch, args
        )
        
        # 验证
        val_results = validate_epoch(
            model, dataloaders['val'], criterion, device, epoch
        )
        
        # 学习率调度
        scheduler.step()
        
        # 日志记录
        logger.info(f"训练损失: {train_results['avg_loss']:.6f}")
        logger.info(f"验证损失: {val_results['val_loss']:.6f}")
        
        # 保存最佳模型
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            save_path = os.path.join(args.save_dir, f"{args.experiment_name}_best.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_results['val_loss'],
                'config': config
            }, save_path)
            logger.info(f"保存最佳模型: {save_path}")
        
        # 定期保存检查点
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"{args.experiment_name}_epoch_{epoch}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_results['val_loss'],
                'config': config
            }, save_path)
    
    # 最终测试
    logger.info("开始最终测试...")
    test_results = validate_epoch(model, dataloaders['test'], criterion, device, config.num_epochs)
    logger.info(f"最终测试结果: {test_results}")
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main() 