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
    """
    Get the image crop for the object specified in the COCO annotations.
    We crop in such a way that in the final resized image, there is `context padding` amount of image data around the object.
    This is the same as is used in RCNN to allow for additional image context.
    """
    # Scale used to compute the new bbox for the image such that there is surrounding context.
    scale = crop_size / (crop_size - padding * 2)

    # Calculate semi-width and semi-height
    semi_width = width / 2
    semi_height = height / 2

    # Calculate the center of the crop
    centerx = x + semi_width
    centery = y + semi_height

    img_width, img_height = img.size

    # We get the crop using the semi- height and width from the center of the crop.
    upper = max(0, centery - (semi_height * scale))
    lower = min(img_height, centery + (semi_height * scale))
    left = max(0, centerx - (semi_width * scale))
    right = min(img_width, centerx + (semi_width * scale))

    crop_img = img.crop((left, upper, right, lower))

    if 0 in crop_img.size:
        print(img.size)
        print("lowx {0}\nlowy {1}\nhighx {2}\nhighy {3}".format(
            left, upper, right, lower))

    return crop_img

class COCOAttributesDataset2017(Dataset):
    """基于成功经验的COCO Attributes 2017数据集"""
    def __init__(self, attributes_file, annotations_file, dataset_root,
                 transforms=None, target_transforms=None,
                 split='train2014', train=True,
                 n_attrs=204, crop_size=224):
        self.attributes_dataset = joblib.load(attributes_file)
        self.coco = COCO(annotations_file)
        self.dataset_root = dataset_root

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.split = split
        self.train = train
        self.n_attrs = n_attrs
        self.crop_size = crop_size

        # 适配数据格式
        logging.info("Loading attributes dataset")
        self.data = []
        
        # 检查数据集格式
        if 'ann_attrs' in self.attributes_dataset:
            # 新格式
            logging.info("Using new format attributes dataset with ann_attrs")
            # 获取所有属性向量
            for ann_id, ann_attr in self.attributes_dataset['ann_attrs'].items():
                if ann_attr['split'] == split:
                    self.data.append(ann_id)
        elif 'ann_vecs' in self.attributes_dataset:
            # 旧格式
            logging.info("Using old format attributes dataset with ann_vecs")
            # 获取所有属性向量
            for patch_id, _ in self.attributes_dataset['ann_vecs'].items():
                if self.attributes_dataset['split'][patch_id] == split:
                    self.data.append(patch_id)
        else:
            raise ValueError("Unknown attributes dataset format")

        # 属性名称列表
        if 'attributes' in self.attributes_dataset:
            self.attributes = sorted(
                self.attributes_dataset['attributes'], key=lambda x: x['id'])
        else:
            # 如果没有属性名称列表，创建一个默认的
            self.attributes = [{'id': i} for i in range(n_attrs)]

        logging.info(f"Loaded {len(self.data)} samples for {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ann_id = self.data[index]

        # 适配新的数据格式
        if 'ann_vecs' in self.attributes_dataset:
            # 旧格式
            attrs = self.attributes_dataset['ann_vecs'][ann_id]
            patch_id_to_ann_id = self.attributes_dataset['patch_id_to_ann_id']
            ann_id_actual = patch_id_to_ann_id[ann_id]
        else:
            # 新格式
            attrs = self.attributes_dataset['ann_attrs'][ann_id]['attrs_vector']
            ann_id_actual = int(ann_id)

        attrs = (attrs > 0).astype(np.float64)

        # coco.loadAnns returns a list
        try:
            ann = self.coco.loadAnns(ann_id_actual)[0]
            image = self.coco.loadImgs(ann['image_id'])[0]
        except:
            # 如果加载失败，返回一个随机样本
            logging.warning(f"Failed to load annotation {ann_id_actual}, using random sample")
            return self.__getitem__((index + 1) % len(self.data))

        x, y, width, height = ann["bbox"]

        # 修改为使用2017版本的路径
        split_dir = 'train2017' if self.split == 'train2014' else 'val2017'
        
        try:
            img = Image.open(os.path.join(self.dataset_root, split_dir,
                                    image["file_name"])).convert('RGB')
        except:
            # 如果加载失败，返回一个随机样本
            logging.warning(f"Failed to load image {image['file_name']}, using random sample")
            return self.__getitem__((index + 1) % len(self.data))

        # Crop out the object with context padding.
        img = get_image_crop(img, x, y, width, height, self.crop_size)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            attrs = self.target_transforms(attrs)

        # 转换为本项目期望的格式
        attrs_tensor = torch.tensor(attrs, dtype=torch.float32)
        
        # 将属性分组为不同类别（基于COCOAttributes的204个属性）
        targets = {
            'color': attrs_tensor[:12],      # 前12个属性作为颜色
            'material': attrs_tensor[12:27], # 接下来15个属性作为材质
            'shape': attrs_tensor[27:47],    # 接下来20个属性作为形状
            'texture': attrs_tensor[47:57],  # 接下来10个属性作为纹理
            'size': attrs_tensor[57:62],     # 接下来5个属性作为大小
            'other': attrs_tensor[62:70]     # 接下来8个属性作为其他
        }

        return img, targets

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
    parser = argparse.ArgumentParser(description='基于成功经验的COCOAttributes训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_coco_success', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs_coco_success', help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true', help='仅进行评估')
    parser.add_argument('--coco_dataset_root', type=str, default='D:/KKK/COCO_Dataset', help='COCO数据集根目录')
    parser.add_argument('--attributes_file', type=str, 
                       default='D:/KKK/jieoulunwen/data/cocottributes-master/cocottributes-master/MSCOCO/cocottributes_new_version.jbl',
                       help='属性数据文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_dir)
    logging.info("开始基于成功经验的COCOAttributes训练...")
    logging.info(f"参数: {args}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    try:
        # 数据变换
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        # 创建数据集
        logging.info("创建数据集...")
        train_dataset = COCOAttributesDataset2017(
            args.attributes_file,
            f"{args.coco_dataset_root}/annotations/instances_train2017.json",
            args.coco_dataset_root,
            transforms=train_transforms,
            split='train2014',
            train=True
        )
        
        val_dataset = COCOAttributesDataset2017(
            args.attributes_file,
            f"{args.coco_dataset_root}/annotations/instances_val2017.json",
            args.coco_dataset_root,
            transforms=val_transforms,
            split='val2014',
            train=False
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # 在Windows上设为0避免多进程问题
            pin_memory=False  # CPU训练时设为False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        logging.info(f"数据加载器创建成功")
        logging.info(f"  训练集: {len(train_dataset)} 个样本")
        logging.info(f"  验证集: {len(val_dataset)} 个样本")
        
        # 导入配置和模型
        from config.base_config import get_config
        config = get_config('COCOAttributes')
        
        # 覆盖配置参数
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        
        logging.info(f"配置加载成功: {config.dataset_name}")
        
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
        
        # 创建优化器和调度器（基于成功经验）
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
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
            val_loss = validate_epoch(model, val_loader, criterion, device, 0)
            logging.info(f"验证损失: {val_loss:.4f}")
            return
        
        # 训练循环
        logging.info(f"开始训练，共 {args.epochs} 个epoch")
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            logging.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            
            # 训练
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config
            )
            
            # 验证
            val_loss = validate_epoch(
                model, val_loader, criterion, device, epoch
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
                best_model_path = os.path.join(args.save_dir, 'best_coco_attributes_success.pth')
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
        print("\n🎉 基于成功经验的COCOAttributes训练完成!")
    else:
        print("\n💥 训练失败!") 