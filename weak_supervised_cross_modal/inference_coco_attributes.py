"""
COCOAttributes数据集推理脚本
"""
import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import argparse
import json
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model(model_path, config, device):
    """加载训练好的模型"""
    from models import WeakSupervisedCrossModalAlignment
    
    model = WeakSupervisedCrossModalAlignment(config)
    
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型训练epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"验证损失: {checkpoint.get('val_loss', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型加载成功")
    else:
        print(f"⚠️ 模型文件不存在，使用随机初始化的模型: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, image_size=224):
    """预处理单张图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor

def predict_attributes(model, image_tensor, config, device):
    """对单张图像进行属性预测"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = outputs['predictions']
        
        # 将logits转换为概率和预测类别
        results = {}
        for attr_name, logits in predictions.items():
            if attr_name in config.num_classes:
                probs = F.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                results[attr_name] = {
                    'predicted_class': pred_class.cpu().item(),
                    'confidence': confidence.cpu().item(),
                    'probabilities': probs.cpu().numpy().tolist()[0]
                }
    
    return results

def get_attribute_class_names():
    """获取属性类别名称映射"""
    class_names = {}
    
    # 颜色属性
    class_names['color'] = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 
        'pink', 'brown', 'black', 'white', 'gray', 'multicolor'
    ]
    
    # 材质属性
    class_names['material'] = [
        'metal', 'wood', 'plastic', 'fabric', 'glass', 'ceramic',
        'leather', 'rubber', 'stone', 'paper', 'fur', 'feather',
        'liquid', 'transparent', 'reflective'
    ]
    
    # 形状属性
    class_names['shape'] = [
        'round', 'square', 'rectangular', 'triangular', 'oval', 'cylindrical',
        'spherical', 'flat', 'curved', 'straight', 'pointed', 'blunt',
        'thin', 'thick', 'long', 'short', 'wide', 'narrow', 'large', 'small'
    ]
    
    # 纹理属性
    class_names['texture'] = [
        'smooth', 'rough', 'bumpy', 'striped', 'spotted', 
        'patterned', 'plain', 'textured', 'shiny', 'matte'
    ]
    
    # 大小属性
    class_names['size'] = [
        'tiny', 'small', 'medium', 'large', 'huge'
    ]
    
    # 其他属性
    class_names['other'] = [
        'natural', 'artificial', 'old', 'new', 'clean', 'dirty', 'broken', 'intact'
    ]
    
    return class_names

def format_results(results, class_names):
    """格式化预测结果"""
    formatted = {}
    
    for attr_name, pred_info in results.items():
        if attr_name in class_names:
            pred_class_idx = pred_info['predicted_class']
            confidence = pred_info['confidence']
            
            if pred_class_idx < len(class_names[attr_name]):
                class_name = class_names[attr_name][pred_class_idx]
            else:
                class_name = f"class_{pred_class_idx}"
            
            formatted[attr_name] = {
                'predicted_class': class_name,
                'confidence': f"{confidence:.4f}",
                'class_index': pred_class_idx
            }
    
    return formatted

def batch_inference(model, image_paths, config, device, batch_size=8):
    """批量推理"""
    results = []
    
    # 预处理所有图像
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img_tensor = preprocess_image(img_path, config.image_size)
            images.append(img_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"跳过无效图像 {img_path}: {e}")
    
    if not images:
        return []
    
    # 批量处理
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_paths = valid_paths[i:i+batch_size]
        
        # 合并为批次
        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            predictions = outputs['predictions']
            
            # 处理批次中的每个样本
            for j in range(batch_tensor.size(0)):
                sample_results = {}
                for attr_name, logits in predictions.items():
                    if attr_name in config.num_classes:
                        probs = F.softmax(logits[j:j+1], dim=-1)
                        pred_class = torch.argmax(probs, dim=-1)
                        confidence = torch.max(probs, dim=-1)[0]
                        
                        sample_results[attr_name] = {
                            'predicted_class': pred_class.cpu().item(),
                            'confidence': confidence.cpu().item(),
                            'probabilities': probs.cpu().numpy().tolist()[0]
                        }
                
                results.append({
                    'image_path': batch_paths[j],
                    'predictions': sample_results
                })
    
    return results

def main():
    """主推理函数"""
    parser = argparse.ArgumentParser(description='COCOAttributes属性预测推理')
    parser.add_argument('--image_path', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录路径（批量推理）')
    parser.add_argument('--model_path', type=str, default='./checkpoints_coco/best_model.pth', help='模型文件路径')
    parser.add_argument('--output_path', type=str, default=None, help='输出结果文件路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批量推理的批次大小')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("错误: 必须指定 --image_path 或 --image_dir")
        return False
    
    print("开始COCOAttributes属性预测推理...")
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    try:
        # 加载配置
        from config.base_config import get_config
        config = get_config('COCOAttributes')
        print(f"配置加载成功")
        
        # 加载模型
        model = load_model(args.model_path, config, device)
        
        # 获取类别名称
        class_names = get_attribute_class_names()
        
        # 单张图像推理
        if args.image_path:
            print(f"\n单张图像推理: {args.image_path}")
            
            if not os.path.exists(args.image_path):
                print(f"错误: 图像文件不存在: {args.image_path}")
                return False
            
            # 预处理图像
            image_tensor = preprocess_image(args.image_path, config.image_size)
            print(f"图像预处理完成: {image_tensor.shape}")
            
            # 进行预测
            results = predict_attributes(model, image_tensor, config, device)
            formatted_results = format_results(results, class_names)
            
            # 打印结果
            print(f"\n📊 预测结果 (图像: {os.path.basename(args.image_path)}):")
            print("=" * 60)
            for attr_name, pred_info in formatted_results.items():
                print(f"{attr_name.upper()}:")
                print(f"  预测类别: {pred_info['predicted_class']}")
                print(f"  置信度: {pred_info['confidence']}")
                print()
            
            # 保存结果
            if args.output_path:
                output_data = {
                    'image_path': args.image_path,
                    'predictions': formatted_results,
                    'raw_results': results
                }
                
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"结果已保存到: {args.output_path}")
        
        # 批量推理
        elif args.image_dir:
            print(f"\n批量推理: {args.image_dir}")
            
            if not os.path.exists(args.image_dir):
                print(f"错误: 目录不存在: {args.image_dir}")
                return False
            
            # 查找图像文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(Path(args.image_dir).glob(f'*{ext}'))
                image_paths.extend(Path(args.image_dir).glob(f'*{ext.upper()}'))
            
            if not image_paths:
                print(f"错误: 在目录中未找到图像文件: {args.image_dir}")
                return False
            
            print(f"找到 {len(image_paths)} 张图像")
            
            # 批量推理
            batch_results = batch_inference(model, image_paths, config, device, args.batch_size)
            
            # 格式化结果
            formatted_batch_results = []
            for result in batch_results:
                formatted_pred = format_results(result['predictions'], class_names)
                formatted_batch_results.append({
                    'image_path': result['image_path'],
                    'predictions': formatted_pred
                })
            
            # 打印部分结果
            print(f"\n📊 批量推理结果 (显示前5个):")
            print("=" * 60)
            for i, result in enumerate(formatted_batch_results[:5]):
                print(f"\n图像 {i+1}: {os.path.basename(result['image_path'])}")
                for attr_name, pred_info in result['predictions'].items():
                    print(f"  {attr_name}: {pred_info['predicted_class']} ({pred_info['confidence']})")
            
            if len(formatted_batch_results) > 5:
                print(f"\n... 还有 {len(formatted_batch_results) - 5} 个结果")
            
            # 保存结果
            if args.output_path:
                output_data = {
                    'total_images': len(formatted_batch_results),
                    'results': formatted_batch_results
                }
                
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n批量结果已保存到: {args.output_path}")
        
        print("\n✅ 推理完成!")
        return True
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 COCOAttributes推理脚本运行成功!")
    else:
        print("\n💥 COCOAttributes推理脚本运行失败!")
