"""
COCOAttributes数据集完整训练脚本
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 创建进度条
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # 移动数据到设备
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets, epoch)
        
        # 检查损失是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"跳过无效损失: {loss.item()}")
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 每100个批次记录一次详细信息
        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f'  {key}: {value.item():.4f}')
    
    return total_loss / max(1, num_batches)

def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # 移动数据到设备
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets, epoch)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
            
            # 更新进度条
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    return total_loss / max(1, num_batches)

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"检查点已保存: {save_path}")

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='COCOAttributes数据集完整训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_coco', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs_coco', help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true', help='仅进行评估')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_dir)
    logging.info("开始COCOAttributes数据集完整训练...")
    logging.info(f"参数: {args}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    try:
        # 导入配置
        from config.base_config import get_config
        config = get_config('COCOAttributes')
        
        # 覆盖配置参数
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        
        logging.info(f"配置加载成功: {config.dataset_name}")
        logging.info(f"属性数量: {config.num_attributes}")
        logging.info(f"属性类别: {config.num_classes}")
        
        # 导入数据适配器
        from data.dataset_adapters import COCOAttributesDatasetAdapter
        adapter = COCOAttributesDatasetAdapter(config)
        dataloaders = adapter.get_dataloaders()
        
        logging.info("数据加载器创建成功")
        for split, dataloader in dataloaders.items():
            logging.info(f"  {split}: {len(dataloader.dataset)} 个样本")
        
        # 导入模型
        from models import WeakSupervisedCrossModalAlignment
        model = WeakSupervisedCrossModalAlignment(config).to(device)
        logging.info("模型创建成功")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        # 导入损失函数
        from training.losses import ComprehensiveLoss
        criterion = ComprehensiveLoss(config)
        logging.info("损失函数创建成功")
        
        # 创建优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 每10个epoch重启一次
            T_mult=2,  # 重启周期倍增
            eta_min=1e-6
        )
        
        logging.info("优化器和调度器创建成功")
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 恢复训练（如果指定）
        start_epoch = 0
        best_val_loss = float('inf')
        
        if args.resume and os.path.exists(args.resume):
            logging.info(f"恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            logging.info(f"从epoch {start_epoch}恢复训练")
        
        # 仅评估模式
        if args.eval_only:
            logging.info("仅进行模型评估...")
            val_loss = validate_epoch(model, dataloaders['val'], criterion, device, 0)
            logging.info(f"验证损失: {val_loss:.4f}")
            return
        
        # 训练循环
        logging.info(f"开始训练，共 {args.epochs} 个epoch")
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            logging.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            
            # 训练
            train_loss = train_epoch(
                model, dataloaders['train'], criterion, optimizer, device, epoch, config
            )
            
            # 验证
            val_loss = validate_epoch(
                model, dataloaders['val'], criterion, device, epoch
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            epoch_time = time.time() - epoch_start_time
            
            logging.info(f"Epoch {epoch + 1} 完成:")
            logging.info(f"  训练损失: {train_loss:.4f}")
            logging.info(f"  验证损失: {val_loss:.4f}")
            logging.info(f"  学习率: {current_lr:.6f}")
            logging.info(f"  耗时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, best_model_path)
                logging.info(f"新的最佳模型! 验证损失: {val_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, checkpoint_path)
        
        logging.info(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
        logging.info(f"模型已保存到: {args.save_dir}")
        logging.info(f"日志已保存到: {log_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 COCOAttributes训练脚本运行成功!")
    else:
        print("\n💥 COCOAttributes训练脚本运行失败!")
