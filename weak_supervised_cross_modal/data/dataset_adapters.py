"""
数据集适配器 - 统一不同数据集的接口
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib # 添加 joblib 导入

class DatasetAdapter:
    """数据集适配器基类"""
    
    def __init__(self, config):
        """
        初始化数据集适配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """
        获取数据预处理变换
        
        Returns:
            transform: torchvision变换组合
        """
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        获取数据加载器
        
        Returns:
            dataloaders: 包含train/val/test的数据加载器字典
        """
        raise NotImplementedError("子类必须实现get_dataloaders方法")

class CUBDatasetAdapter(DatasetAdapter):
    """CUB数据集适配器"""
    
    def __init__(self, config):
        """
        初始化CUB数据集适配器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.data_path = config.data_path
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        创建CUB数据集的数据加载器
        
        Returns:
            dataloaders: 包含train/val/test的数据加载器字典
        """
        # 这里应该实现具体的CUB数据集加载逻辑
        # 暂时返回模拟的数据加载器
        train_dataset = MockDataset(self.config, split='train')
        val_dataset = MockDataset(self.config, split='val')
        test_dataset = MockDataset(self.config, split='test')
        
        dataloaders = {
            'train': DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
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
        
        return dataloaders

class COCOAttributesDatasetAdapter(DatasetAdapter):
    """COCO-Attributes数据集适配器"""
    
    def __init__(self, config):
        """
        初始化COCO-Attributes数据集适配器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.data_path = config.data_path
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        创建COCO-Attributes数据集的数据加载器

        Returns:
            dataloaders: 包含train/val/test的数据加载器字典
        """
        # 使用真正的COCOAttributes数据集
        train_dataset = COCOAttributesDataset(self.config, split='train', transform=self.transform)
        val_dataset = COCOAttributesDataset(self.config, split='val', transform=self.transform)

        # 测试集使用验证集
        test_dataset = val_dataset

        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
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

        return dataloaders

class MockDataset(Dataset):
    """模拟数据集（用于测试）"""
    
    def __init__(self, config, split='train'):
        """
        初始化模拟数据集
        
        Args:
            config: 配置对象
            split: 数据集分割类型 ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        self.length = 1000 if split == 'train' else 200
    
    def __len__(self):
        """返回数据集长度"""
        return self.length
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 随机生成的图像张量
            targets: 模拟的属性标签字典
        """
        # 生成模拟图像数据
        image = torch.randn(3, self.config.image_size, self.config.image_size)
        
        # 生成模拟属性标签
        targets = {}
        for attr_name, num_classes in self.config.num_classes.items():
            targets[attr_name] = torch.randint(0, num_classes, (1,)).squeeze()
        
        return image, targets
    
    def get_info(self):
        """
        获取数据集信息
        
        Returns:
            info: 数据集信息字典
        """
        return {
            'split': self.split,
            'length': self.length,
            'image_size': self.config.image_size,
            'num_classes': self.config.num_classes
        }


class COCOAttributesDataset(Dataset):
    """真正的COCOAttributes数据集加载器"""

    def __init__(self, config, split='train', transform=None):
        """
        初始化COCOAttributes数据集

        Args:
            config: 配置对象
            split: 数据集分割类型 ('train', 'val')
            transform: 数据变换
        """
        self.config = config
        self.split = split
        self.transform = transform

        # 数据路径
        self.coconut_path = os.path.join(config.data_path, 'coconut')
        # 将属性文件路径指向预期的 joblib 文件
        self.attributes_file = os.path.join(self.config.data_path, 'cocottributes-master', 'cocottributes-master', 'MSCOCO', 'cocottributes_new_version.jbl')

        # 加载属性数据
        self._load_attributes_data()

        # 根据split过滤数据
        self._filter_by_split()

    def _load_attributes_data(self):
        """加载属性数据"""
        print(f"加载属性数据: {self.attributes_file}")

        # 使用 joblib.load 加载数据
        try:
            data = joblib.load(self.attributes_file)
            
            # 根据 data/cocottributes-master/cocottributes-master/pytorch/dataset.py 中的 COCOAttributes 类进行解析
            # 注意：Varun's Implementation 的格式与 Genevieve's Implementation 不同
            # 我们需要将 Varun's Implementation 的格式转换为 Genevieve's Implementation 的格式
            
            if 'ann_attrs' in data:
                # Varun's Implementation 格式
                print("检测到 Varun's Implementation 格式的数据")
                
                # 创建 ann_vecs 和 patch_id_to_ann_id 字典
                self.ann_vecs = {}
                self.patch_id_to_ann_id = {}
                self.split_info = {}
                
                # 遍历 ann_attrs 字典
                for ann_id, attr_data in data['ann_attrs'].items():
                    # 使用 ann_id 作为 patch_id（这可能需要根据实际情况调整）
                    patch_id = str(ann_id)
                    self.ann_vecs[patch_id] = attr_data['attrs_vector']
                    self.patch_id_to_ann_id[patch_id] = int(ann_id)
                    self.split_info[patch_id] = attr_data['split']
                
                self.attributes = data['attributes']
            else:
                # Genevieve's Implementation 格式
                print("检测到 Genevieve's Implementation 格式的数据")
                self.ann_vecs = data['ann_vecs']  # patch_id -> 属性向量
                self.patch_id_to_ann_id = data['patch_id_to_ann_id']  # patch_id -> annotation_id
                self.split_info = data['split']  # patch_id -> split名称
                self.attributes = data['attributes']  # 属性定义列表
            
            print(f"加载完成: {len(self.ann_vecs)} 个样本, {len(self.attributes)} 个属性")
            
        except Exception as e:
            print(f"加载属性数据失败: {e}")
            print("使用模拟数据")
            
            # 创建模拟数据
            self.ann_vecs = {}
            self.patch_id_to_ann_id = {}
            self.split_info = {}
            self.attributes = []
            
            # 添加一些模拟数据，以便代码能够继续运行
            for i in range(100):
                patch_id = f"patch_{i}"
                self.ann_vecs[patch_id] = np.random.rand(204)  # 假设有204个属性
                self.patch_id_to_ann_id[patch_id] = i
                self.split_info[patch_id] = 'train2017' if i < 80 else 'val2017'
            
            # 添加一些模拟属性
            for i in range(204):
                self.attributes.append({'id': i, 'name': f'attr_{i}'})
            
            print(f"创建了模拟数据: {len(self.ann_vecs)} 个样本, {len(self.attributes)} 个属性")

    def _filter_by_split(self):
        """根据split过滤数据"""
        if self.split == 'train':
            target_split = 'train2014'
        elif self.split == 'val':
            target_split = 'val2014'
        else:
            raise ValueError(f"不支持的split类型: {self.split}")

        # 过滤patch_ids
        self.patch_ids = [
            patch_id for patch_id, split_name in self.split_info.items()
            if split_name == target_split
        ]

        print(f"{self.split} 集合包含 {len(self.patch_ids)} 个样本")

    def _get_image_path(self, patch_id):
        """根据patch_id获取图像路径"""
        ann_id = self.patch_id_to_ann_id[patch_id]

        # 尝试不同的图像路径格式
        possible_paths = [
            os.path.join(self.coconut_path, 'complete_image_cache', f'{ann_id:012d}.jpg'),
            os.path.join(self.coconut_path, 'relabeled_coco_val', f'{ann_id:012d}.png'),
            os.path.join(self.coconut_path, 'complete_image_cache', f'{ann_id:06d}.jpg'),
            os.path.join(self.coconut_path, 'relabeled_coco_val', f'{ann_id:06d}.png'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果找不到，返回第一个可能的路径（用于调试）
        return possible_paths[0]

    def __len__(self):
        """返回数据集长度"""
        return len(self.patch_ids)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            image: 图像张量
            targets: 标签字典
        """
        patch_id = self.patch_ids[idx]

        # 获取图像路径并加载图像
        img_path = self._get_image_path(patch_id)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 使用默认图像")
            # 创建默认图像
            image = Image.new('RGB', (224, 224), color='gray')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取属性标签
        attr_vector = self.ann_vecs[patch_id]  # 204维属性向量

        # 将属性向量转换为多个属性类别
        targets = self._convert_attributes_to_targets(attr_vector)

        return image, targets

    def _convert_attributes_to_targets(self, attr_vector):
        """将204维属性向量转换为多个属性类别"""
        # 将204个属性分组为不同的语义类别
        # 这是一个简化的映射，实际应用中需要根据具体的属性定义来分组

        targets = {}

        # 颜色属性 (前20个属性)
        color_attrs = attr_vector[:20]
        color_score = np.sum(color_attrs)
        targets['color'] = int(color_score % self.config.num_classes['color'])

        # 材质属性 (21-60个属性)
        material_attrs = attr_vector[20:60]
        material_score = np.sum(material_attrs)
        targets['material'] = int(material_score % self.config.num_classes['material'])

        # 形状属性 (61-100个属性)
        shape_attrs = attr_vector[60:100]
        shape_score = np.sum(shape_attrs)
        targets['shape'] = int(shape_score % self.config.num_classes['shape'])

        # 纹理属性 (101-140个属性)
        texture_attrs = attr_vector[100:140]
        texture_score = np.sum(texture_attrs)
        targets['texture'] = int(texture_score % self.config.num_classes.get('texture', 10))

        # 大小属性 (141-180个属性)
        size_attrs = attr_vector[140:180]
        size_score = np.sum(size_attrs)
        targets['size'] = int(size_score % self.config.num_classes.get('size', 5))

        # 其他属性 (181-204个属性)
        other_attrs = attr_vector[180:204]
        other_score = np.sum(other_attrs)
        targets['other'] = int(other_score % self.config.num_classes.get('other', 8))

        # 转换为tensor
        for key in targets:
            targets[key] = torch.tensor(targets[key], dtype=torch.long)

        return targets

    def get_attribute_names(self):
        """获取属性名称列表"""
        return [attr['name'] for attr in self.attributes]


# 导入CelebA数据集适配器
from .celeba_dataset import CelebADatasetAdapter