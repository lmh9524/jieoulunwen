"""
测试COCOAttributes数据集加载器
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config.base_config import get_config
from data.dataset_adapters import COCOAttributesDatasetAdapter

def test_coco_attributes_dataset():
    """测试COCOAttributes数据集加载"""
    print("开始测试COCOAttributes数据集加载器...")
    
    # 获取配置
    config = get_config('COCOAttributes')
    print(f"数据路径: {config.data_path}")
    print(f"图像大小: {config.image_size}")
    print(f"属性数量: {config.num_attributes}")
    print(f"属性类别: {config.num_classes}")
    
    # 创建数据适配器
    try:
        adapter = COCOAttributesDatasetAdapter(config)
        print("✅ COCOAttributes数据适配器创建成功")
    except Exception as e:
        print(f"❌ COCOAttributes数据适配器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 创建数据加载器
    try:
        dataloaders = adapter.get_dataloaders()
        print("✅ 数据加载器创建成功")
        
        # 检查数据加载器
        for split, dataloader in dataloaders.items():
            print(f"  {split}: {len(dataloader.dataset)} 个样本")
            
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试加载一个批次的数据
    try:
        train_loader = dataloaders['train']
        print(f"\n测试数据加载 (批次大小: {train_loader.batch_size})...")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"✅ 成功加载第 {batch_idx + 1} 个批次:")
            print(f"  图像形状: {images.shape}")
            print(f"  图像数据类型: {images.dtype}")
            print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            
            print(f"  目标键: {list(targets.keys())}")
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: 形状={value.shape}, 类型={value.dtype}, 范围=[{value.min()}, {value.max()}]")
                else:
                    print(f"    {key}: {type(value)}")
            
            # 只测试第一个批次
            break
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试验证集
    try:
        val_loader = dataloaders['val']
        print(f"\n测试验证集数据加载...")
        
        for batch_idx, (images, targets) in enumerate(val_loader):
            print(f"✅ 验证集批次 {batch_idx + 1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  目标数量: {len(targets)}")
            break
            
    except Exception as e:
        print(f"❌ 验证集数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ COCOAttributes数据集测试完成!")
    return True

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n开始测试模型兼容性...")
    
    try:
        # 获取配置
        config = get_config('COCOAttributes')
        
        # 创建数据加载器
        adapter = COCOAttributesDatasetAdapter(config)
        dataloaders = adapter.get_dataloaders()
        
        # 导入模型
        from models import WeakSupervisedCrossModalAlignment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = WeakSupervisedCrossModalAlignment(config).to(device)
        print(f"✅ 模型创建成功，设备: {device}")
        
        # 测试前向传播
        train_loader = dataloaders['train']
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = model(images)
            
            print(f"✅ 前向传播成功:")
            print(f"  输入形状: {images.shape}")
            print(f"  输出键: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if isinstance(value, dict):
                    print(f"    {key}: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"      {sub_key}: {sub_value.shape}")
                elif isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
            
            break
        
        print("✅ 模型兼容性测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 模型兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n开始测试损失计算...")
    
    try:
        # 获取配置
        config = get_config('COCOAttributes')
        
        # 创建数据加载器
        adapter = COCOAttributesDatasetAdapter(config)
        dataloaders = adapter.get_dataloaders()
        
        # 创建模型和损失函数
        from models import WeakSupervisedCrossModalAlignment
        from training.losses import ComprehensiveLoss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = WeakSupervisedCrossModalAlignment(config).to(device)
        criterion = ComprehensiveLoss(config)
        
        print("✅ 模型和损失函数创建成功")
        
        # 测试损失计算
        train_loader = dataloaders['train']
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            total_loss, loss_dict = criterion(outputs, targets, epoch=0)
            
            print(f"✅ 损失计算成功:")
            print(f"  总损失: {total_loss.item():.4f}")
            print(f"  损失组件:")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.item():.4f}")
                else:
                    print(f"    {key}: {value}")
            
            break
        
        print("✅ 损失计算测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始COCOAttributes完整测试")
    print("=" * 60)
    
    # 测试1: 数据集加载
    success1 = test_coco_attributes_dataset()
    
    # 测试2: 模型兼容性
    success2 = test_model_compatibility()
    
    # 测试3: 损失计算
    success3 = test_loss_computation()
    
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    print(f"  数据集加载: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"  模型兼容性: {'✅ 成功' if success2 else '❌ 失败'}")
    print(f"  损失计算: {'✅ 成功' if success3 else '❌ 失败'}")
    
    if success1 and success2 and success3:
        print("\n🎉 所有测试通过! COCOAttributes数据集已准备就绪!")
        print("\n下一步可以运行:")
        print("  python train_coco_attributes.py --epochs 5")
        return True
    else:
        print("\n💥 部分测试失败! 需要进一步调试!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 COCOAttributes测试完成!")
    else:
        print("\n🔧 COCOAttributes需要进一步调试!")
