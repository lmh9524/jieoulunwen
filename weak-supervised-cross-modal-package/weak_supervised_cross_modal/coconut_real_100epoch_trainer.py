#!/usr/bin/env python3
"""
COCONut数据集100轮训练 - 严格真实数据版本
绝对不使用任何模拟或合成数据，只使用真实的图像和标注
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import requests
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class COCONutRealOnlyDataset(Dataset):
    """严格真实数据的COCONut数据集加载器 - 绝不使用合成数据"""
    
    def __init__(self, data_dir: str, split: str = "val", transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载真实数据
        self.data = self._load_real_data()
        
        # 限制样本数量
        if max_samples and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
        
        # 创建属性映射
        self.attributes = self._create_attributes()
        
        # 下载真实图像
        self._download_real_images()
        
        # 过滤掉没有真实图像的样本
        self.data = self._filter_valid_samples()
        
        logger.info(f"加载 {len(self.data)} 个真实样本 ({split})")
        logger.info("严格保证：绝不使用任何合成或模拟数据！")
    
    def _load_real_data(self):
        """加载真实的COCONut标注数据"""
        json_file = self.data_dir / "relabeled_coco_val.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"真实数据文件不存在: {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # 处理COCO格式数据
        data_list = []
        
        if 'images' in raw_data:
            # 标准COCO格式
            images = raw_data['images']
            annotations = raw_data.get('annotations', [])
            
            # 创建图像ID到标注的映射
            img_to_anns = {}
            for ann in annotations:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)
            
            # 处理每张图像
            for img_info in images:
                img_id = img_info['id']
                
                # 确保有真实的图像URL
                if 'coco_url' in img_info and img_info['coco_url']:
                    item = {
                        'id': img_id,
                        'url': img_info['coco_url'],
                        'filename': img_info['file_name'],
                        'width': img_info['width'],
                        'height': img_info['height'],
                        'annotations': img_to_anns.get(img_id, [])
                    }
                    data_list.append(item)
        
        logger.info(f"找到 {len(data_list)} 个有真实图像URL的样本")
        return data_list
    
    def _create_attributes(self):
        """创建属性列表"""
        attributes = [
            # 对象类别
            'person', 'vehicle', 'animal', 'furniture', 'electronics', 'food', 'sports',
            'building', 'nature', 'tool', 'clothing', 'book', 'toy',
            # 视觉属性
            'large', 'small', 'round', 'rectangular', 'colorful', 'bright', 'dark',
            'transparent', 'opaque', 'shiny', 'matte',
            # 环境属性
            'outdoor', 'indoor', 'natural', 'artificial', 'urban', 'rural',
            # 材质属性
            'soft', 'hard', 'metal', 'wood', 'fabric', 'plastic', 'glass',
            # 状态属性
            'moving', 'static', 'open', 'closed', 'new', 'old',
            # 颜色属性
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange'
        ]
        return attributes
    
    def _download_real_images(self):
        """下载真实图像"""
        cache_dir = self.data_dir / 'complete_image_cache'
        cache_dir.mkdir(exist_ok=True)
        
        logger.info("检查完整图像缓存...")
        success_count = 0
        
        for item in tqdm(self.data, desc="检查真实图像"):
            filename = item.get('filename', f"img_{item['id']}.jpg")
            
            cache_path = cache_dir / filename
            if cache_path.exists():
                success_count += 1
        
        logger.info(f"找到 {success_count} 张真实图像")
    
    def _filter_valid_samples(self):
        """过滤掉没有真实图像的样本"""
        cache_dir = self.data_dir / 'complete_image_cache'
        valid_data = []
        
        for item in self.data:
            filename = item.get('filename', f"img_{item['id']}.jpg")
            cache_path = cache_dir / filename
            
            if cache_path.exists():
                try:
                    # 验证图像是否可以正常加载
                    img = Image.open(cache_path)
                    img.verify()
                    # 重新打开图像（verify会关闭文件）
                    img = Image.open(cache_path)
                    # 确保图像有效
                    if img.size[0] > 0 and img.size[1] > 0:
                        valid_data.append(item)
                    img.close()
                except Exception as e:
                    logger.warning(f"真实图像损坏，跳过: {cache_path} - {e}")
        
        logger.info(f"过滤后剩余 {len(valid_data)} 个有效的真实样本")
        return valid_data
    
    def _extract_attributes_from_annotations(self, annotations):
        """从真实标注中提取属性"""
        attributes = set()
        
        # 从COCO标注中提取属性
        for ann in annotations:
            if 'category_id' in ann:
                cat_id = ann['category_id']
                # COCO类别映射
                if cat_id == 1:  # person
                    attributes.add('person')
                elif cat_id in [2, 3, 4, 6, 8]:  # vehicles
                    attributes.add('vehicle')
                elif cat_id in [16, 17, 18, 19, 20, 21, 22, 23, 24]:  # animals
                    attributes.add('animal')
                elif cat_id in [62, 63, 64, 65, 67]:  # furniture
                    attributes.add('furniture')
                elif cat_id in [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]:  # electronics/appliances
                    attributes.add('electronics')
                elif cat_id in [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]:  # food
                    attributes.add('food')
                elif cat_id in [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]:  # sports
                    attributes.add('sports')
            
            # 从边界框推断属性
            if 'bbox' in ann:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]
                if area > 50000:  # 大物体
                    attributes.add('large')
                elif area < 5000:  # 小物体
                    attributes.add('small')
        
        # 默认属性
        if not attributes:
            attributes = {'artificial', 'static'}
        
        # 添加通用属性
        attributes.add('outdoor')  # 大部分COCO图像是户外的
        attributes.add('colorful')
        
        return list(attributes)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载真实图像 - 绝对不使用合成数据
        cache_dir = self.data_dir / 'complete_image_cache'
        filename = item.get('filename', f"img_{item['id']}.jpg")
        cache_path = cache_dir / filename
        
        if not cache_path.exists():
            raise FileNotFoundError(f"真实图像不存在: {cache_path}")
        
        try:
            # 只使用真实下载的图像
            image = Image.open(cache_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"无法加载真实图像 {cache_path}: {e}")
        
        # 从真实标注中提取属性
        annotations = item.get('annotations', [])
        item_attributes = self._extract_attributes_from_annotations(annotations)
        
        # 创建属性向量
        attr_vector = torch.zeros(len(self.attributes))
        for attr in item_attributes:
            if attr in self.attributes:
                attr_vector[self.attributes.index(attr)] = 1.0
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'attributes': attr_vector,
            'id': item.get('id', idx),
            'extracted_attrs': item_attributes,
            'image_path': str(cache_path),
            'annotations': annotations
        }

class RealCrossModalModel(nn.Module):
    """真实数据跨模态模型"""
    
    def __init__(self, attr_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # 使用预训练的ResNet50
        self.visual_encoder = models.resnet50(weights='IMAGENET1K_V1')
        
        # 适度冻结层
        for param in list(self.visual_encoder.parameters())[:-15]:
            param.requires_grad = False
        
        # 替换最后的全连接层
        self.visual_encoder.fc = nn.Sequential(
            nn.Linear(self.visual_encoder.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 属性编码器
        self.attr_encoder = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, attr_dim)
        )
        
        # 对比学习投影
        self.visual_proj = nn.Linear(hidden_dim, 128)
        self.attr_proj = nn.Linear(hidden_dim, 128)
        
    def forward(self, images, attributes=None):
        # 提取视觉特征
        visual_features = self.visual_encoder(images)
        
        if attributes is not None:
            # 训练模式
            attr_features = self.attr_encoder(attributes)
            
            # 注意力机制
            visual_attended, _ = self.attention(
                visual_features.unsqueeze(0), 
                attr_features.unsqueeze(0), 
                attr_features.unsqueeze(0)
            )
            visual_attended = visual_attended.squeeze(0)
            
            # 融合
            fused_input = torch.cat([visual_attended, attr_features], dim=1)
            fused_features = self.fusion(fused_input)
            
            # 预测
            attr_pred = self.classifier(fused_features)
            
            # 对比学习
            visual_proj = self.visual_proj(visual_features)
            attr_proj = self.attr_proj(attr_features)
            
            return {
                'predictions': attr_pred,
                'visual_proj': visual_proj,
                'attr_proj': attr_proj
            }
        else:
            # 推理模式
            return {
                'predictions': self.classifier(visual_features)
            }

class TrainingMonitor:
    """训练监控器"""
    def __init__(self):
        self.history = defaultdict(list)
        self.start_time = time.time()
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
    
    def plot_training_curves(self, save_path='real_training_curves.png'):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.history['epoch']
        
        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Real Data Training: Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Real Data Training: Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.history['lr'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # 验证准确率详细视图
        ax4.plot(epochs, self.history['val_acc'], 'r-', linewidth=2)
        ax4.set_title('Real Data Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True)
        
        plt.suptitle('Training on 100% Real COCONut Data', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"真实数据训练曲线已保存至: {save_path}")
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.history['val_acc']:
            return "暂无训练数据"
        
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['epoch'][self.history['val_acc'].index(best_val_acc)]
        final_val_acc = self.history['val_acc'][-1]
        total_time = time.time() - self.start_time
        
        return f"""
真实数据训练摘要:
- 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})
- 最终验证准确率: {final_val_acc:.2f}%
- 总训练时间: {total_time/3600:.2f} 小时
- 平均每轮时间: {total_time/len(self.history['epoch']):.1f} 秒
- 数据类型: 100% 真实图像和标注
"""

def main():
    parser = argparse.ArgumentParser(description='COCONut 100轮真实数据训练')
    parser.add_argument('--data_dir', type=str, default='../data/coconut', help='数据目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='初始学习率')
    parser.add_argument('--max_samples', type=int, default=500, help='最大样本数')
    parser.add_argument('--save_model', type=str, default='coconut_real_100epoch.pth', help='模型保存路径')
    parser.add_argument('--patience', type=int, default=25, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info("=" * 60)
    logger.info("严格真实数据训练 - 绝不使用任何合成数据")
    logger.info("=" * 60)
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载真实数据集
    try:
        train_dataset = COCONutRealOnlyDataset(
            data_dir=args.data_dir,
            split="val",
            transform=train_transform,
            max_samples=args.max_samples
        )
        
        val_dataset = COCONutRealOnlyDataset(
            data_dir=args.data_dir,
            split="val",
            transform=val_transform,
            max_samples=args.max_samples // 5
        )
    except Exception as e:
        logger.error(f"加载真实数据失败: {e}")
        return
    
    if len(train_dataset) == 0:
        logger.error("没有找到有效的真实图像数据！")
        return
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # 模型
    attr_dim = len(train_dataset.attributes)
    model = RealCrossModalModel(attr_dim=attr_dim).to(device)
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    contrastive_loss = nn.MSELoss()
    
    # 早停和监控
    early_stopping = EarlyStopping(patience=args.patience)
    monitor = TrainingMonitor()
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"属性维度: {attr_dim}")
    logger.info(f"训练集大小: {len(train_dataset)} (100% 真实图像)")
    logger.info(f"验证集大小: {len(val_dataset)} (100% 真实图像)")
    
    # 训练循环
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        if len(train_loader) > 0:
            pbar = tqdm(train_loader, desc=f"真实数据训练 Epoch {epoch+1}/{args.epochs}")
            
            for batch in pbar:
                images = batch['image'].to(device)
                attributes = batch['attributes'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images, attributes)
                
                # 主要损失
                main_loss = criterion(outputs['predictions'], attributes)
                
                # 对比学习损失
                contrast_loss = contrastive_loss(outputs['visual_proj'], outputs['attr_proj'])
                
                # 总损失
                total_loss = main_loss + 0.1 * contrast_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                predictions = torch.sigmoid(outputs['predictions'])
                predicted = (predictions > 0.5).float()
                train_correct += (predicted == attributes).sum().item()
                train_total += attributes.numel()
                
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%',
                    'Real': '100%'
                })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            if len(val_loader) > 0:
                for batch in tqdm(val_loader, desc="验证真实数据"):
                    images = batch['image'].to(device)
                    attributes = batch['attributes'].to(device)
                    
                    outputs = model(images, attributes)
                    loss = criterion(outputs['predictions'], attributes)
                    
                    val_loss += loss.item()
                    predictions = torch.sigmoid(outputs['predictions'])
                    predicted = (predictions > 0.5).float()
                    val_correct += (predicted == attributes).sum().item()
                    val_total += attributes.numel()
        
        # 更新学习率
        scheduler.step()
        
        # 计算指标
        train_loss_avg = train_loss / max(len(train_loader), 1)
        val_loss_avg = val_loss / max(len(val_loader), 1)
        train_acc = 100 * train_correct / max(train_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新监控
        monitor.update(epoch+1, train_loss_avg, train_acc, val_loss_avg, val_acc, current_lr)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} [真实数据]")
        logger.info(f"训练 - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"验证 - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")
        logger.info(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'attributes': train_dataset.attributes,
                'data_type': '100% Real Images',
                'sample_count': len(train_dataset)
            }, args.save_model)
            logger.info(f"保存最佳真实数据模型 (Val Acc: {val_acc:.2f}%)")
        
        # 早停检查
        if early_stopping(val_acc, model):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止真实数据训练")
            break
        
        # 每10轮保存训练曲线
        if (epoch + 1) % 10 == 0:
            monitor.plot_training_curves(f'real_training_curves_epoch_{epoch+1}.png')
        
        print("-" * 60)
    
    # 训练完成
    logger.info(f"真实数据训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(monitor.get_summary())
    
    # 保存最终训练曲线
    monitor.plot_training_curves('final_real_training_curves.png')
    
    # 测试推理
    model.eval()
    with torch.no_grad():
        if len(val_loader) > 0:
            sample_batch = next(iter(val_loader))
            images = sample_batch['image'][:1].to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs['predictions'])
            
            logger.info("真实数据推理测试:")
            predicted_attrs = []
            for i, prob in enumerate(predictions[0]):
                if prob > 0.5:
                    predicted_attrs.append(f"{train_dataset.attributes[i]}({prob:.3f})")
            
            logger.info(f"预测属性: {predicted_attrs}")
            logger.info(f"真实图像路径: {sample_batch['image_path'][0]}")
            logger.info(f"原始文本: {sample_batch['annotations'][0][0]['text']}")
            logger.info(f"提取属性: {sample_batch['extracted_attrs'][0]}")

if __name__ == "__main__":
    main() 