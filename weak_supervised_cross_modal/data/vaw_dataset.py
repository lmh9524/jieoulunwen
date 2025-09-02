"""
VAW数据集加载器 - 简化版本
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

class VAWDataset(Dataset):
    """VAW数据集加载器"""
    
    def __init__(self, annotations_dir: str, images_dir: str, split: str = "train", 
                 transform=None, max_samples=None):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.data = self._load_annotations()
        
        # 限制样本数量
        if max_samples and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
        
        # 构建属性词典
        self.attribute_vocab = self._build_attribute_vocab()
        
        print(f"VAW数据集加载: {split}, {len(self.data)} 样本, {len(self.attribute_vocab)} 属性")
    
    def _load_annotations(self):
        """加载标注文件"""
        if self.split == "train":
            files = ["train_part1.json"]  # 简化：只用第一部分
        else:
            files = [f"{self.split}.json"]
        
        all_data = []
        for filename in files:
            filepath = self.annotations_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_data.extend(data)
        
        return all_data
    
    def _build_attribute_vocab(self):
        """构建属性词典"""
        all_attributes = set()
        for sample in self.data:
            if "positive_attributes" in sample:
                all_attributes.update(sample["positive_attributes"])
            if "negative_attributes" in sample:
                all_attributes.update(sample["negative_attributes"])
        
        return sorted(list(all_attributes))
    
    def _encode_attributes(self, positive_attrs, negative_attrs):
        """编码属性为向量"""
        attr_vector = torch.full((len(self.attribute_vocab),), 0.5)  # 中性值
        
        attr_to_idx = {attr: idx for idx, attr in enumerate(self.attribute_vocab)}
        
        for attr in positive_attrs:
            if attr in attr_to_idx:
                attr_vector[attr_to_idx[attr]] = 1.0
        
        for attr in negative_attrs:
            if attr in attr_to_idx:
                attr_vector[attr_to_idx[attr]] = 0.0
        
        return attr_vector
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 创建默认图像（因为我们只有少量样本图像）
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # 编码属性
        attributes = self._encode_attributes(
            sample.get("positive_attributes", []),
            sample.get("negative_attributes", [])
        )
        
        return {
            "images": image,
            "attributes": attributes,
            "metadata": {
                "image_id": sample["image_id"],
                "object_name": sample["object_name"]
            }
        }

class VAWDatasetAdapter:
    """VAW数据集适配器"""
    
    def __init__(self, config):
        self.config = config
    
    def get_transforms(self, split):
        """获取数据变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self):
        """创建数据加载器"""
        dataloaders = {}
        
        for split in ["train", "val", "test"]:
            dataset = VAWDataset(
                annotations_dir=self.config.vaw_annotations_dir,
                images_dir=self.config.vaw_images_dir,
                split=split,
                transform=self.get_transforms(split),
                max_samples=getattr(self.config, f"max_{split}_samples", None)
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == "train"),
                num_workers=0,  # 简化：不使用多进程
                drop_last=(split == "train")
            )
            
            dataloaders[split] = dataloader
        
        return dataloaders

