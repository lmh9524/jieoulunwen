"""
基于真实COCO 2017数据集和cocottributes属性标注的训练脚本
使用275个高质量的object-level属性样本进行训练
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import logging
import argparse
from tqdm import tqdm
import time
from datetime import datetime
from pycocotools.coco import COCO

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'coco_real_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

class COCORealAttributesDataset(Dataset):
    """基于真实COCO 2017数据的属性数据集"""
    
    def __init__(self, mapping_file, split='train', transform=None):
        """
        Args:
            mapping_file: coco2017_instances_attributes_mapping.json文件路径
            split: 'train' 或 'val'
            transform: 图像变换
        """
        self.split = split
        self.transform = transform
        
        # 加载映射数据
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        # 筛选指定split的数据
        if split == 'train':
            # 重新划分：前220个作为训练集
            self.samples = mapping_data['data'][:220]
        else:
            # 后55个作为验证集
            self.samples = mapping_data['data'][220:]
        
        self.num_attributes = mapping_data['attributes_info']['total_attributes']
        
        logging.info(f"加载{split}数据集: {len(self.samples)}个样本, {self.num_attributes}个属性")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image_path = sample['file_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.warning(f"无法加载图像 {image_path}: {e}")
            # 创建默认图像
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 根据bbox裁剪目标区域
        bbox = sample['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # 确保bbox在图像范围内
        img_width, img_height = image.size
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        # 裁剪目标区域
        cropped_image = image.crop((x, y, x + w, y + h))
        
        # 应用变换
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # 获取属性标签
        attrs_vector = np.array(sample['attrs_vector'], dtype=np.float32)
        attrs_tensor = torch.from_numpy(attrs_vector)
        
        return cropped_image, attrs_tensor, sample['ann_id']

class AttributeClassifier(nn.Module):
    """多标签属性分类器"""
    
    def __init__(self, num_attributes=204, backbone='resnet50'):
        super(AttributeClassifier, self).__init__()
        
        # 使用预训练的ResNet作为backbone
        if backbone == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # 移除最后的分类层
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 多标签分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_attributes)
        )
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 分类
        logits = self.classifier(features)
        
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (images, targets, ann_ids) in enumerate(progress_bar):
        # 移动数据到设备
        images = images.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 用于计算精度的统计
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
        
        for batch_idx, (images, targets, ann_ids) in enumerate(progress_bar):
            # 移动数据到设备
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 计算预测精度（使用0.5阈值）
            predictions = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predictions == targets.bool()).float().sum().item()
            total_predictions += targets.numel()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'Accuracy': f'{accuracy:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = correct_predictions / total_predictions
    
    return avg_loss, avg_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train COCO Real Attributes Model')
    parser.add_argument('--mapping_file', type=str, 
                      default='coco2017_instances_attributes_mapping.json',
                      help='映射文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--image_size', type=int, default=224, help='图像大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_coco_real', 
                      help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs_coco_real', 
                      help='日志保存目录')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_dir)
    logging.info(f"开始训练 - 日志文件: {log_file}")
    logging.info(f"参数: {args}")
    
    # 检查映射文件
    if not os.path.exists(args.mapping_file):
        logging.error(f"映射文件不存在: {args.mapping_file}")
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COCORealAttributesDataset(
        args.mapping_file, split='train', transform=transform)
    val_dataset = COCORealAttributesDataset(
        args.mapping_file, split='val', transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=2, pin_memory=True)
    
    logging.info(f"训练集: {len(train_dataset)} 样本")
    logging.info(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = AttributeClassifier(num_attributes=train_dataset.num_attributes)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logging.info("开始训练...")
    
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logging.info(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        logging.info(f"训练损失: {train_loss:.4f}")
        logging.info(f"验证损失: {val_loss:.4f}")
        logging.info(f"验证精度: {val_accuracy:.4f}")
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': args
            }
            
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logging.info(f"保存最佳模型: {best_model_path} (验证精度: {val_accuracy:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': args
            }, checkpoint_path)
            logging.info(f"保存检查点: {checkpoint_path}")
    
    logging.info(f"\n训练完成!")
    logging.info(f"最佳验证精度: {best_val_accuracy:.4f}")
    logging.info(f"模型保存在: {args.save_dir}")
    logging.info(f"日志文件: {log_file}")

if __name__ == "__main__":
    main() 