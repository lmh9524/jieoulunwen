"""
测试基于真实COCO 2017数据训练的cocottributes属性模型
"""
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import numpy as np
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AttributeClassifier(nn.Module):
    """多标签属性分类器（与训练时保持一致）"""
    
    def __init__(self, num_attributes=204, backbone='resnet50'):
        super(AttributeClassifier, self).__init__()
        
        # 使用预训练的ResNet作为backbone
        if backbone == 'resnet50':
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

def load_model(checkpoint_path, num_attributes=204):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = AttributeClassifier(num_attributes=num_attributes)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功 - Epoch: {checkpoint['epoch']}, 验证精度: {checkpoint['val_accuracy']:.4f}")
    
    return model, device

def test_model_on_samples(mapping_file, model_path, num_test_samples=10):
    """在测试样本上评估模型"""
    
    # 加载映射数据
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # 使用验证集样本进行测试
    test_samples = mapping_data['data'][220:220+num_test_samples]  # 取前10个验证样本
    
    # 加载模型
    model, device = load_model(model_path, mapping_data['attributes_info']['total_attributes'])
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n测试 {len(test_samples)} 个样本...")
    print("=" * 80)
    
    total_correct = 0
    total_predictions = 0
    
    for i, sample in enumerate(test_samples):
        # 加载图像
        image_path = sample['file_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"样本 {i+1}: 无法加载图像 {image_path}")
            continue
        
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
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = torch.sigmoid(outputs) > 0.5
        
        # 真实标签
        true_labels = torch.tensor(sample['attrs_vector'], dtype=torch.bool).to(device)
        
        # 计算精度
        correct = (predictions.squeeze() == true_labels).float().sum().item()
        total = true_labels.numel()
        accuracy = correct / total
        
        total_correct += correct
        total_predictions += total
        
        # 统计预测的属性数量
        pred_positive = predictions.sum().item()
        true_positive = true_labels.sum().item()
        
        print(f"样本 {i+1}: ann_id={sample['ann_id']}, category_id={sample['category_id']}")
        print(f"  精度: {accuracy:.4f} ({correct:.0f}/{total})")
        print(f"  预测正属性: {pred_positive}, 真实正属性: {true_positive}")
        print(f"  bbox: {bbox}")
        print()
    
    # 总体统计
    overall_accuracy = total_correct / total_predictions
    print("=" * 80)
    print(f"总体精度: {overall_accuracy:.4f} ({total_correct:.0f}/{total_predictions})")
    
    return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description='Test COCO Real Attributes Model')
    parser.add_argument('--mapping_file', type=str, 
                      default='coco2017_instances_attributes_mapping.json',
                      help='映射文件路径')
    parser.add_argument('--model_path', type=str, 
                      default='./checkpoints_coco_real/best_model.pth',
                      help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=10, 
                      help='测试样本数量')
    
    args = parser.parse_args()
    
    # 检查文件存在性
    if not os.path.exists(args.mapping_file):
        print(f"错误: 映射文件不存在: {args.mapping_file}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 测试模型
    accuracy = test_model_on_samples(args.mapping_file, args.model_path, args.num_samples)
    
    print(f"\n测试完成，总体精度: {accuracy:.4f}")

if __name__ == "__main__":
    main() 