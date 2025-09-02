"""
VAW训练器 - 修复版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict

class VAWTrainer:
    """VAW训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        # 创建目录
        Path("logs/vaw_training").mkdir(parents=True, exist_ok=True)
        Path("checkpoints/vaw_training").mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = defaultdict(list)
        
        print("VAW训练器初始化完成")
    
    def create_dataloaders(self):
        """创建数据加载器"""
        from ..data.vaw_dataset import VAWDatasetAdapter
        adapter = VAWDatasetAdapter(self.config)
        return adapter.get_dataloaders()
    
    def create_model(self):
        """创建模型"""
        from ..models.vaw_model import create_vaw_model
        return create_vaw_model(self.config).to(self.device)
    
    def train_epoch(self, model, dataloader, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch["images"].to(self.device)
            attributes = batch["attributes"].to(self.device)
            
            # 前向传播
            outputs = model(images, attributes)
            
            # 计算损失
            attr_pred = outputs["attribute_predictions"]
            loss = nn.BCEWithLogitsLoss()(attr_pred, attributes)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, model, dataloader):
        """验证模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                attributes = batch["attributes"].to(self.device)
                
                outputs = model(images, attributes)
                attr_pred = outputs["attribute_predictions"]
                
                loss = nn.BCEWithLogitsLoss()(attr_pred, attributes)
                total_loss += loss.item()
                
                # 计算准确率
                pred_binary = (torch.sigmoid(attr_pred) > 0.5).float()
                correct += (pred_binary == attributes).float().mean().item()
                total += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(self):
        """完整训练流程"""
        print("开始VAW训练...")
        
        # 创建组件
        dataloaders = self.create_dataloaders()
        model = self.create_model()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f"训练设备: {self.device}")
        print(f"训练轮数: {self.config.num_epochs}")
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(model, dataloaders["train"], optimizer)
            
            # 验证
            val_results = self.validate(model, dataloaders["val"])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results[\"loss\"]:.4f}")
            print(f"Val Acc: {val_results[\"accuracy\"]:.4f}")
            
            # 保存最佳模型
            if val_results["accuracy"] > self.best_metric:
                self.best_metric = val_results["accuracy"]
                torch.save(model.state_dict(), "checkpoints/vaw_training/best_model.pth")
                print(f"保存最佳模型，准确率: {self.best_metric:.4f}")
        
        print(f"\\n训练完成！最佳准确率: {self.best_metric:.4f}")
        return model

