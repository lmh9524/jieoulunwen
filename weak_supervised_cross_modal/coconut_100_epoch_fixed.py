#!/usr/bin/env python3
"""
COCONut数据集100轮Epoch训练脚本 - 修复版本
包含优化的训练策略：学习率调度、早停、模型保存等
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

class COCONutStableDataset(Dataset):
    """稳定版COCONut数据集加载器"""
    
    def __init__(self, data_dir: str, split: str = "val", transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.data = self._load_data()
        
        # 确保有足够的样本
        self._ensure_sufficient_samples(max_samples)
        
        # 创建属性映射
        self.attributes = self._create_attributes()
        
        logger.info(f"加载 {len(self.data)} 个样本 ({split})")
    
    def _load_data(self):
        """加载真实的COCONut数据"""
        json_file = self.data_dir / f"relabeled_coco_val.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # 处理数据格式
        data_list = []
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, dict):
                    value['id'] = key
                    data_list.append(value)
                else:
                    item = {
                        'id': key,
                        'url': value if isinstance(value, str) and value.startswith('http') else None,
                        'text': value if isinstance(value, str) and not value.startswith('http') else '',
                        'width': 640,
                        'height': 480
                    }
                    data_list.append(item)
        
        return data_list
    
    def _ensure_sufficient_samples(self, max_samples):
        """确保有足够的样本"""
        if len(self.data) < 50:  # 如果样本太少，复制数据
            original_data = self.data.copy()
            target_size = max_samples if max_samples else 500
            while len(self.data) < target_size:
                self.data.extend(original_data)
                if len(self.data) >= target_size:
                    break
        
        # 限制样本数量
        if max_samples and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
    
    def _create_attributes(self):
        """创建属性列表"""
        attributes = [
            # 对象类别
            'person', 'vehicle', 'animal', 'furniture', 'electronics', 'food', 'sports',
            # 视觉属性
            'large', 'small', 'round', 'rectangular', 'colorful', 'bright', 'dark',
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
    
    def _extract_attributes_from_text(self, text):
        """从文本提取属性"""
        if not text:
            return ['artificial', 'static', 'colorful']
        
        text = text.lower()
        attributes = []
        
        # 对象检测
        if any(word in text for word in ['person', 'man', 'woman', 'people']):
            attributes.append('person')
        if any(word in text for word in ['car', 'vehicle', 'truck', 'bus']):
            attributes.append('vehicle')
        if any(word in text for word in ['animal', 'dog', 'cat', 'bird']):
            attributes.append('animal')
        if any(word in text for word in ['furniture', 'chair', 'table', 'bed']):
            attributes.append('furniture')
        if any(word in text for word in ['phone', 'computer', 'tv', 'electronics']):
            attributes.append('electronics')
        if any(word in text for word in ['food', 'fruit', 'drink', 'meal']):
            attributes.append('food')
        if any(word in text for word in ['sport', 'ball', 'game']):
            attributes.append('sports')
        
        # 尺寸
        if any(word in text for word in ['large', 'big', 'huge']):
            attributes.append('large')
        elif any(word in text for word in ['small', 'little', 'tiny']):
            attributes.append('small')
        
        # 形状
        if any(word in text for word in ['round', 'circular']):
            attributes.append('round')
        elif any(word in text for word in ['rectangular', 'square']):
            attributes.append('rectangular')
        
        # 环境
        if any(word in text for word in ['outdoor', 'outside', 'park']):
            attributes.append('outdoor')
        elif any(word in text for word in ['indoor', 'inside', 'room']):
            attributes.append('indoor')
        
        # 颜色
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange']
        for color in colors:
            if color in text:
                attributes.append(color)
        
        # 默认属性
        if not attributes:
            attributes = ['artificial', 'static', 'colorful']
        
        return list(set(attributes))
    
    def _create_synthetic_image(self, text, item_id):
        """创建合成图像"""
        # 基于文本生成颜色
        if 'red' in text.lower():
            color = (255, 100, 100)
        elif 'blue' in text.lower():
            color = (100, 100, 255)
        elif 'green' in text.lower():
            color = (100, 255, 100)
        else:
            # 基于ID生成一致的颜色
            hash_val = hash(str(item_id)) % 1000000
            color = (
                (hash_val % 256),
                ((hash_val // 256) % 256),
                ((hash_val // 65536) % 256)
            )
        
        # 创建基础图像
        image = Image.new('RGB', (224, 224), color=color)
        
        # 添加纹理
        img_array = np.array(image)
        seed = abs(hash(str(item_id))) % (2**32 - 1)
        noise = np.random.RandomState(seed).randint(-20, 20, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 创建合成图像
        text = item.get('text', '')
        image = self._create_synthetic_image(text, item.get('id', idx))
        
        # 提取属性
        item_attributes = self._extract_attributes_from_text(text)
        
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
            'text': text,
            'id': item.get('id', idx),
            'extracted_attrs': item_attributes
        }

class StableCrossModalModel(nn.Module):
    """稳定版跨模态模型"""
    
    def __init__(self, attr_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # 使用预训练的ResNet50
        self.visual_encoder = models.resnet50(pretrained=True)
        
        # 冻结前面的层
        for param in list(self.visual_encoder.parameters())[:-10]:
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
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.history['epoch']
        
        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
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
        ax4.set_title('Validation Accuracy (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存至: {save_path}")
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.history['val_acc']:
            return "暂无训练数据"
        
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['epoch'][self.history['val_acc'].index(best_val_acc)]
        final_val_acc = self.history['val_acc'][-1]
        total_time = time.time() - self.start_time
        
        return f"""
训练摘要:
- 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})
- 最终验证准确率: {final_val_acc:.2f}%
- 总训练时间: {total_time/3600:.2f} 小时
- 平均每轮时间: {total_time/len(self.history['epoch']):.1f} 秒
"""

def main():
    parser = argparse.ArgumentParser(description='COCONut 100轮Epoch训练 - 稳定版')
    parser.add_argument('--data_dir', type=str, default='../data/coconut', help='数据目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大样本数')
    parser.add_argument('--save_model', type=str, default='coconut_100epoch_stable.pth', help='模型保存路径')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = COCONutStableDataset(
        data_dir=args.data_dir,
        split="val",
        transform=train_transform,
        max_samples=args.max_samples
    )
    
    val_dataset = COCONutStableDataset(
        data_dir=args.data_dir,
        split="val",
        transform=val_transform,
        max_samples=args.max_samples // 5
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows兼容性
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
    model = StableCrossModalModel(attr_dim=attr_dim).to(device)
    
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
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 训练循环
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        if len(train_loader) > 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
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
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            if len(val_loader) > 0:
                for batch in tqdm(val_loader, desc="Validating"):
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
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
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
                'attributes': train_dataset.attributes
            }, args.save_model)
            logger.info(f"保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        # 早停检查
        if early_stopping(val_acc, model):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
        
        # 每10轮保存训练曲线
        if (epoch + 1) % 10 == 0:
            monitor.plot_training_curves(f'training_curves_epoch_{epoch+1}.png')
        
        print("-" * 60)
    
    # 训练完成
    logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(monitor.get_summary())
    
    # 保存最终训练曲线
    monitor.plot_training_curves('final_training_curves.png')
    
    # 测试推理
    model.eval()
    with torch.no_grad():
        if len(val_loader) > 0:
            sample_batch = next(iter(val_loader))
            images = sample_batch['image'][:1].to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs['predictions'])
            
            logger.info("推理测试:")
            predicted_attrs = []
            for i, prob in enumerate(predictions[0]):
                if prob > 0.5:
                    predicted_attrs.append(f"{train_dataset.attributes[i]}({prob:.3f})")
            
            logger.info(f"预测属性: {predicted_attrs}")
            logger.info(f"原始文本: {sample_batch['text'][0]}")
            logger.info(f"提取属性: {sample_batch['extracted_attrs'][0]}")

if __name__ == "__main__":
    main() 