"""
CUB数据集推理脚本
"""
import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import argparse
import json

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model(model_path, config):
    """加载训练好的模型"""
    from models import WeakSupervisedCrossModalAlignment
    
    model = WeakSupervisedCrossModalAlignment(config)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"模型加载成功: {model_path}")
    else:
        print(f"模型文件不存在，使用随机初始化的模型: {model_path}")
    
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
    model.to(device)
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

def get_class_names():
    """获取CUB数据集的类别名称"""
    # 这里可以加载真实的类别名称映射
    # 暂时使用简化的映射
    class_names = {}
    
    # 颜色类别（示例）
    class_names['color'] = [
        'black', 'blue', 'brown', 'buff', 'gray', 'green', 
        'orange', 'pink', 'purple', 'red', 'white'
    ]
    
    # 材质类别（示例）
    class_names['material'] = [
        'solid', 'striped', 'spotted', 'mottled', 'iridescent', 
        'metallic', 'glossy', 'matte'
    ]
    
    # 形状类别（示例）
    class_names['shape'] = [
        'round', 'oval', 'elongated', 'curved', 'straight', 'pointed',
        'broad', 'narrow', 'thick', 'thin', 'large', 'small', 'medium',
        'triangular', 'rectangular'
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

def main():
    """主推理函数"""
    parser = argparse.ArgumentParser(description='CUB数据集属性预测推理')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth', help='模型文件路径')
    parser.add_argument('--output_path', type=str, default=None, help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    print("开始CUB数据集属性预测推理...")
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    try:
        # 加载配置
        from config.base_config import get_config
        config = get_config('CUB')
        print("配置加载成功")
        
        # 加载模型
        model = load_model(args.model_path, config)
        
        # 预处理图像
        if not os.path.exists(args.image_path):
            print(f"图像文件不存在: {args.image_path}")
            return False
        
        image_tensor = preprocess_image(args.image_path, config.image_size)
        print(f"图像预处理完成: {image_tensor.shape}")

        # 进行预测
        results = predict_attributes(model, image_tensor, config, device)
        print("属性预测完成")
        
        # 获取类别名称并格式化结果
        class_names = get_class_names()
        formatted_results = format_results(results, class_names)
        
        # 打印结果
        print(f"\n预测结果 (图像: {os.path.basename(args.image_path)}):")
        print("=" * 50)
        for attr_name, pred_info in formatted_results.items():
            print(f"{attr_name.upper()}:")
            print(f"  预测类别: {pred_info['predicted_class']}")
            print(f"  置信度: {pred_info['confidence']}")
            print()
        
        # 保存结果到文件
        if args.output_path:
            output_data = {
                'image_path': args.image_path,
                'predictions': formatted_results,
                'raw_results': results
            }
            
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {args.output_path}")

        print("推理完成!")
        return True

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n推理脚本运行成功!")
    else:
        print("\n推理脚本运行失败!")
