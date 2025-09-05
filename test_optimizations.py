#!/usr/bin/env python3
"""
优化修正验证脚本 - 测试所有修正是否正常工作
"""

import sys
sys.path.append('./weak_supervised_cross_modal')

def test_enhanced_transforms():
    """测试增强的数据变换"""
    print("🧪 测试增强数据变换...")
    
    try:
        from config.celeba_optimized_config import get_optimized_config
        from data.celeba_optimized_dataset import CelebAOptimizedDatasetAdapter
        
        config = get_optimized_config(1)
        config.data_path = 'D:\\KKK\\data\\CelebA'
        
        adapter = CelebAOptimizedDatasetAdapter(config)
        transforms = adapter._get_transforms()
        
        print("  ✅ 训练变换包含以下增强:")
        for i, transform in enumerate(transforms['train'].transforms):
            print(f"    {i+1}. {transform.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据变换测试失败: {e}")
        return False

def test_focal_loss():
    """测试Focal Loss"""
    print("\n🧪 测试Focal Loss...")
    
    try:
        from training.focal_loss import FocalLoss, create_adaptive_loss
        import torch
        
        # 测试基础Focal Loss
        focal_loss = FocalLoss(gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        
        loss = focal_loss(inputs, targets)
        print(f"  ✅ Focal Loss计算: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Focal Loss测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🔬 优化修正验证测试")
    print("=" * 50)
    
    tests = [
        ("增强数据变换", test_enhanced_transforms),
        ("Focal Loss", test_focal_loss)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有优化修正验证通过!")
    else:
        print("⚠️ 部分测试失败")
    
    return passed == total

if __name__ == "__main__":
    main() 