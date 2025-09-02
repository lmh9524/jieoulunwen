"""
基于成功训练经验的COCOAttributes训练脚本
结合之前在cocottributes-master上的成功配置
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import argparse
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import joblib
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO

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

def get_image_crop(img, x, y, width, height, crop_size=224, padding=16):
    """基于成功经验的图像裁剪函数"""
    scale = crop_size / (crop_size - padding * 2)
    semi_width = width / 2
    semi_height = height / 2
    centerx = x + semi_width
    centery = y + semi_height
    img_width, img_height = img.size
    
    upper = max(0, centery - (semi_height * scale))
    lower = min(img_height, centery + (semi_height * scale))
    left = max(0, centerx - (semi_width * scale))
    right = min(img_width, centerx + (semi_width * scale))
    
    crop_img = img.crop((left, upper, right, lower))
    return crop_img

class COCOAttributesSuccessDataset(Dataset):
    """基于成功经验的COCO Attributes数据集"""
    def __init__(self, attributes_file, annotations_file, dataset_root,
                 transforms=None, split='train2014', n_attrs=204):
        self.attributes_dataset = joblib.load(attributes_file)
        self.coco = COCO(annotations_file)
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.split = split
        self.n_attrs = n_attrs
        
        # 加载数据
        self.data = []
        if 'ann_attrs' in self.attributes_dataset:
            logging.info("Using new format attributes dataset with ann_attrs")
            for ann_id, ann_attr in self.attributes_dataset['ann_attrs'].items():
                if ann_attr['split'] == split:
                    self.data.append(ann_id)
        
        logging.info(f"Loaded {len(self.data)} samples for {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ann_id = self.data[index]
        
        # 获取属性向量
        attrs = self.attributes_dataset['ann_attrs'][ann_id]['attrs_vector']
        ann_id_actual = int(ann_id)
        attrs = (attrs > 0).astype(np.float64)
        
        try:
            ann = self.coco.loadAnns(ann_id_actual)[0]
            image = self.coco.loadImgs(ann['image_id'])[0]
            x, y, width, height = ann["bbox"]
            
            # 使用2017版本路径
            split_dir = 'train2017' if self.split == 'train2014' else 'val2017'
            img = Image.open(os.path.join(self.dataset_root, split_dir, image["file_name"])).convert('RGB')
            img = get_image_crop(img, x, y, width, height, 224)
            
            if self.transforms:
                img = self.transforms(img)
            
            # 转换为项目需要的格式
            attrs_tensor = torch.tensor(attrs, dtype=torch.float32)
            targets = {
                'color': attrs_tensor[:12],
                'material': attrs_tensor[12:27],
                'shape': attrs_tensor[27:47],
                'texture': attrs_tensor[47:57],
                'size': attrs_tensor[57:62],
                'other': attrs_tensor[62:70]
            }
            
            return img, targets
            
        except Exception as e:
            # 错误处理，返回随机样本
            return self.__getitem__((index + 1) % len(self.data))

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='基于成功经验的COCOAttributes训练')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_success', help='保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs_success', help='日志目录')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_dir)
    logging.info("开始基于成功经验的COCOAttributes训练...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    try:
        # 创建数据变换
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = COCOAttributesSuccessDataset(
            'D:/KKK/jieoulunwen/data/cocottributes-master/cocottributes-master/MSCOCO/cocottributes_new_version.jbl',
            'D:/KKK/COCO_Dataset/annotations/instances_train2017.json',
            'D:/KKK/COCO_Dataset',
            transforms=train_transforms,
            split='train2014'
        )
        
        val_dataset = COCOAttributesSuccessDataset(
            'D:/KKK/jieoulunwen/data/cocottributes-master/cocottributes-master/MSCOCO/cocottributes_new_version.jbl',
            'D:/KKK/COCO_Dataset/annotations/instances_val2017.json',
            'D:/KKK/COCO_Dataset',
            transforms=val_transforms,
            split='val2014'
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        logging.info(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
        
        # 导入模型和配置
        from config.base_config import get_config
        from models import WeakSupervisedCrossModalAlignment
        from training.losses import ComprehensiveLoss
        
        config = get_config('COCOAttributes')
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        
        model = WeakSupervisedCrossModalAlignment(config).to(device)
        criterion = ComprehensiveLoss(config)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        # 训练循环
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            for images, targets in progress_bar:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                optimizer.zero_grad()
                outputs = model(images)
                loss, _ = criterion(outputs, targets, epoch)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
                
                avg_loss = train_loss / max(1, num_batches)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = {k: v.to(device) for k, v in targets.items()}
                    
                    outputs = model(images)
                    loss, _ = criterion(outputs, targets, epoch)
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        val_batches += 1
            
            scheduler.step()
            
            avg_train_loss = train_loss / max(1, num_batches)
            avg_val_loss = val_loss / max(1, val_batches)
            
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, os.path.join(args.save_dir, 'best_coco_attributes_success.pth'))
                logging.info(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")
        
        logging.info(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
        return True
        
    except Exception as e:
        logging.error(f"训练失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 训练成功完成!")
    else:
        print("💥 训练失败!") 