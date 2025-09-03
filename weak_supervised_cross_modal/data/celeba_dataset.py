"""
CelebA数据集适配器
支持40个面部属性的加载和处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class CelebADataset(Dataset):
    """CelebA数据集加载器"""
    
    def __init__(self, config, split='train', transform=None):
        """
        初始化CelebA数据集
        
        Args:
            config: 配置对象
            split: 数据集分割类型 ('train', 'val', 'test')
            transform: 数据变换
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # 数据路径
        self.data_root = config.data_path
        self.images_dir = os.path.join(self.data_root, 'img_align_celeba')
        self.annotations_dir = os.path.join(self.data_root, 'Anno')
        
        # 加载数据
        self._load_annotations()
        self._filter_by_split()
        
        print(f"CelebA {split} 数据集加载完成: {len(self.image_files)} 个样本")
    
    def _load_annotations(self):
        """加载CelebA标注文件"""
        # 加载属性标注
        attr_file = os.path.join(self.annotations_dir, 'list_attr_celeba.txt')
        self.attr_df = pd.read_csv(attr_file, sep=r'\s+', header=1, index_col=0)
        
        # 加载数据集分割信息 (在Eval目录中)
        eval_dir = os.path.join(self.data_root, 'Eval')
        partition_file = os.path.join(eval_dir, 'list_eval_partition.txt')
        self.partition_df = pd.read_csv(partition_file, sep=r'\s+', header=None, 
                                       names=['filename', 'partition'], index_col=0)
        
        # CelebA的40个属性名称
        self.attribute_names = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
            'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
            'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        
        print(f"加载了 {len(self.attr_df)} 个样本的属性标注")
        print(f"属性数量: {len(self.attribute_names)}")
    
    def _filter_by_split(self):
        """根据split过滤数据"""
        # CelebA官方分割: 0=train, 1=val, 2=test
        if self.split == 'train':
            target_partition = 0
        elif self.split == 'val':
            target_partition = 1
        elif self.split == 'test':
            target_partition = 2
        else:
            raise ValueError(f"不支持的split类型: {self.split}")
        
        # 获取对应分割的文件名
        split_filenames = self.partition_df[
            self.partition_df['partition'] == target_partition
        ].index.tolist()
        
        # 过滤属性数据
        self.split_attr_df = self.attr_df.loc[split_filenames]
        self.image_files = split_filenames
        
        print(f"{self.split} 分割包含 {len(self.image_files)} 个样本")
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 图像张量
            targets: 标签字典
        """
        filename = self.image_files[idx]
        
        # 加载图像
        img_path = os.path.join(self.images_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 使用默认图像")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取属性标签
        attr_labels = self.split_attr_df.loc[filename].values  # 40个属性的值 (-1或1)
        
        # 将属性标签转换为目标格式
        targets = self._convert_attributes_to_targets(attr_labels)
        
        return image, targets
    
    def _convert_attributes_to_targets(self, attr_labels):
        """
        将40个CelebA属性转换为模型需要的目标格式
        
        Args:
            attr_labels: 40维属性标签数组 (值为-1或1)
            
        Returns:
            targets: 标签字典
        """
        # 将-1转换为0，1保持为1 (二分类)
        binary_labels = (attr_labels == 1).astype(int)
        
        targets = {}
        
        # 将40个属性分组为不同的语义类别用于解耦训练
        
        # 头发相关属性 (索引: 4, 5, 8, 9, 11, 17, 28, 32, 33)
        hair_attrs = [4, 5, 8, 9, 11, 17, 28, 32, 33]  # Bald, Bangs, Black_Hair, etc.
        hair_score = np.sum(binary_labels[hair_attrs])
        targets['hair_style'] = torch.tensor(hair_score % self.config.num_classes.get('hair_style', 8), dtype=torch.long)
        
        # 面部特征 (索引: 1, 3, 6, 7, 13, 14, 19, 23, 25, 26, 27, 29)
        facial_attrs = [1, 3, 6, 7, 13, 14, 19, 23, 25, 26, 27, 29]
        facial_score = np.sum(binary_labels[facial_attrs])
        targets['facial_features'] = torch.tensor(facial_score % self.config.num_classes.get('facial_features', 10), dtype=torch.long)
        
        # 化妆相关 (索引: 18, 36, 37)
        makeup_attrs = [18, 36, 37]  # Heavy_Makeup, Wearing_Lipstick, Wearing_Necklace
        makeup_score = np.sum(binary_labels[makeup_attrs])
        targets['makeup'] = torch.tensor(makeup_score % self.config.num_classes.get('makeup', 4), dtype=torch.long)
        
        # 配饰 (索引: 15, 34, 35, 38)
        accessory_attrs = [15, 34, 35, 38]  # Eyeglasses, Wearing_Earrings, etc.
        accessory_score = np.sum(binary_labels[accessory_attrs])
        targets['accessories'] = torch.tensor(accessory_score % self.config.num_classes.get('accessories', 5), dtype=torch.long)
        
        # 表情相关 (索引: 21, 31)
        expression_attrs = [21, 31]  # Mouth_Slightly_Open, Smiling
        expression_score = np.sum(binary_labels[expression_attrs])
        targets['expression'] = torch.tensor(expression_score % self.config.num_classes.get('expression', 3), dtype=torch.long)
        
        # 性别和年龄 (索引: 20, 39)
        demo_attrs = [20, 39]  # Male, Young
        demo_score = np.sum(binary_labels[demo_attrs])
        targets['demographics'] = torch.tensor(demo_score % self.config.num_classes.get('demographics', 4), dtype=torch.long)
        
        # 胡须相关 (索引: 0, 16, 22, 24, 30)
        beard_attrs = [0, 16, 22, 24, 30]  # 5_o_Clock_Shadow, Goatee, etc.
        beard_score = np.sum(binary_labels[beard_attrs])
        targets['facial_hair'] = torch.tensor(beard_score % self.config.num_classes.get('facial_hair', 6), dtype=torch.long)
        
        # 吸引力和图像质量 (索引: 2, 10)
        quality_attrs = [2, 10]  # Attractive, Blurry
        quality_score = np.sum(binary_labels[quality_attrs])
        targets['quality'] = torch.tensor(quality_score % self.config.num_classes.get('quality', 3), dtype=torch.long)
        
        # 将原始40维属性也包含进去（用于完整的属性预测）
        targets['all_attributes'] = torch.tensor(binary_labels, dtype=torch.float32)
        
        return targets
    
    def get_attribute_names(self):
        """获取属性名称列表"""
        return self.attribute_names
    
    def get_split_info(self):
        """获取数据分割信息"""
        return {
            'split': self.split,
            'total_samples': len(self.image_files),
            'attributes_count': len(self.attribute_names)
        }


class CelebADatasetAdapter:
    """CelebA数据集适配器"""
    
    def __init__(self, config):
        """
        初始化CelebA数据集适配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """获取数据预处理变换"""
        train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return {'train': train_transform, 'val': val_transform, 'test': val_transform}
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        创建CelebA数据集的数据加载器
        
        Returns:
            dataloaders: 包含train/val/test的数据加载器字典
        """
        # 创建数据集
        train_dataset = CelebADataset(
            self.config, 
            split='train', 
            transform=self.transform['train']
        )
        val_dataset = CelebADataset(
            self.config, 
            split='val', 
            transform=self.transform['val']
        )
        test_dataset = CelebADataset(
            self.config, 
            split='test', 
            transform=self.transform['test']
        )
        
        # 创建数据加载器
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        }
        
        print("CelebA数据加载器创建完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本") 
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return dataloaders 