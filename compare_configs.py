#!/usr/bin/env python3
"""
配置对比工具 - 对比baseline与优化配置的变化
"""

import sys
sys.path.append('./weak_supervised_cross_modal')

from config.base_config import get_config
from config.celeba_optimized_config import get_optimized_config
import json

def compare_attribute_groups():
    """对比属性分组变化"""
    baseline_config = get_config('CelebA')
    optimized_config = get_optimized_config(1)
    
    print("=" * 60)
    print("属性分组对比分析")
    print("=" * 60)
    
    print("\n📊 Baseline配置:")
    baseline_total = 0
    for attr, num_classes in baseline_config.num_classes.items():
        print(f"  {attr}: {num_classes}类")
        baseline_total += num_classes
    print(f"  总分类数: {baseline_total}")
    
    print("\n🎯 优化配置:")
    optimized_total = 0
    for attr, num_classes in optimized_config.num_classes.items():
        print(f"  {attr}: {num_classes}类")
        optimized_total += num_classes
    print(f"  总分类数: {optimized_total}")
    
    print(f"\n📈 变化:")
    print(f"  分类数变化: {baseline_total} → {optimized_total} ({optimized_total - baseline_total:+d})")
    
    # 分组变化分析
    print("\n🔄 属性分组变化:")
    baseline_attrs = set(baseline_config.num_classes.keys())
    optimized_attrs = set(optimized_config.num_classes.keys())
    
    removed = baseline_attrs - optimized_attrs
    added = optimized_attrs - baseline_attrs
    unchanged = baseline_attrs & optimized_attrs
    
    if removed:
        print(f"  ❌ 移除: {list(removed)}")
    if added:
        print(f"  ✅ 新增: {list(added)}")
    if unchanged:
        print(f"  ⚪ 保持: {list(unchanged)}")

def compare_loss_weights():
    """对比损失权重变化"""
    baseline_config = get_config('CelebA')
    optimized_config = get_optimized_config(1)
    
    print("\n" + "=" * 60)
    print("损失权重对比分析")
    print("=" * 60)
    
    # 获取所有权重键
    all_keys = set(baseline_config.loss_weights.keys()) | set(optimized_config.loss_weights.keys())
    
    print(f"{'权重名称':<20} {'Baseline':<10} {'优化版':<10} {'变化':<15} {'说明'}")
    print("-" * 70)
    
    changes = []
    for key in sorted(all_keys):
        baseline_val = baseline_config.loss_weights.get(key, 0.0)
        optimized_val = optimized_config.loss_weights.get(key, 0.0)
        
        if baseline_val != optimized_val:
            change = optimized_val - baseline_val
            change_str = f"{change:+.3f}"
            if change > 0:
                explanation = "⬆️ 增强"
            elif change < 0:
                explanation = "⬇️ 降低"
            else:
                explanation = "➡️ 不变"
            
            changes.append((key, baseline_val, optimized_val, change))
            print(f"{key:<20} {baseline_val:<10.3f} {optimized_val:<10.3f} {change_str:<15} {explanation}")
    
    print(f"\n📊 权重调整统计:")
    print(f"  调整项目数: {len(changes)}")
    increases = sum(1 for _, _, _, change in changes if change > 0)
    decreases = sum(1 for _, _, _, change in changes if change < 0)
    print(f"  权重增加: {increases}项")
    print(f"  权重降低: {decreases}项")

def compare_training_config():
    """对比训练配置变化"""
    baseline_config = get_config('CelebA')
    optimized_config = get_optimized_config(1)
    
    print("\n" + "=" * 60)
    print("训练配置对比分析")
    print("=" * 60)
    
    configs = [
        ('batch_size', '批处理大小'),
        ('learning_rate', '学习率'),
        ('num_epochs', '训练轮数'),
    ]
    
    print(f"{'配置项':<15} {'Baseline':<15} {'优化版':<15} {'变化'}")
    print("-" * 60)
    
    for attr, desc in configs:
        baseline_val = getattr(baseline_config, attr)
        optimized_val = getattr(optimized_config, attr)
        
        if baseline_val != optimized_val:
            if isinstance(baseline_val, float):
                change = f"{optimized_val/baseline_val:.2f}x"
            else:
                change = f"{optimized_val - baseline_val:+d}"
        else:
            change = "无变化"
            
        print(f"{desc:<15} {baseline_val:<15} {optimized_val:<15} {change}")
    
    # 新增配置
    print(f"\n🆕 新增配置:")
    new_configs = [
        ('early_stopping_patience', '早停容忍'),
        ('lr_reduce_patience', 'LR调度容忍'),
        ('lr_reduce_factor', 'LR衰减因子'),
    ]
    
    for attr, desc in new_configs:
        if hasattr(optimized_config, attr):
            val = getattr(optimized_config, attr)
            print(f"  {desc}: {val}")

def show_stage_progression():
    """显示阶段渐进配置"""
    print("\n" + "=" * 60)
    print("阶段渐进配置")
    print("=" * 60)
    
    for stage in [1, 2, 3]:
        config = get_optimized_config(stage)
        print(f"\n🎯 Stage {stage}: {config.dataset_name}")
        print(f"  训练轮数: {config.num_epochs}")
        
        # 启用的模块
        modules = []
        if config.use_frequency_decoupling:
            modules.append("AFANet")
        if config.use_hierarchical_decomposition:
            modules.append("WINNER")
        if config.use_dynamic_routing:
            modules.append("MAVD")
        if config.use_cmdl_regularization:
            modules.append("CMDL")
        
        print(f"  启用模块: {modules if modules else ['基础分类']}")
        
        # 主要损失权重
        key_weights = ['reg', 'cal', 'hierarchy', 'frequency', 'mavd']
        active_weights = {k: v for k, v in config.loss_weights.items() 
                         if k in key_weights and v > 0}
        if active_weights:
            print(f"  激活权重: {active_weights}")
        else:
            print(f"  激活权重: 仅分类损失")

def estimate_improvements():
    """预估改进效果"""
    print("\n" + "=" * 60)
    print("预估改进效果")
    print("=" * 60)
    
    # 基于测试集分析的baseline性能
    baseline_performance = {
        'hair_style': 52.72,      # 问题属性 → hair_basic (简化)
        'facial_features': 34.49, # 最差属性 → face_structure + face_details (拆分)
        'makeup': 78.91,          # 保持
        'accessories': 86.99,     # 保持
        'expression': 87.71,      # 保持
        'demographics': 87.31,    # 保持
        'facial_hair': 90.03,     # 保持
        'quality': 79.66,         # 保持
        'mean_accuracy': 74.73
    }
    
    # 预估改进后性能
    print("📊 性能预估分析:")
    print(f"{'属性组':<20} {'Baseline':<10} {'预估改进':<10} {'提升':<10} {'策略'}")
    print("-" * 70)
    
    estimated_improvements = []
    
    # 重点改进项
    improvements = [
        ('hair_basic', 52.72, 65.0, '简化分类(8→5类)'),
        ('face_structure', 34.49, 55.0, '拆分facial_features'),
        ('face_details', 34.49, 50.0, '拆分facial_features'),
        ('expression', 87.71, 88.0, '权重微调'),
        ('demographics', 87.31, 87.5, '权重微调'),
        ('makeup', 78.91, 80.0, '权重微调'),
        ('accessories', 86.99, 87.2, '权重微调'),
        ('facial_hair', 90.03, 89.5, '平衡精确率召回率'),
        ('quality', 79.66, 80.0, '权重微调'),
    ]
    
    total_baseline = 0
    total_estimated = 0
    
    for attr, baseline, estimated, strategy in improvements:
        improvement = estimated - baseline
        total_baseline += baseline
        total_estimated += estimated
        
        print(f"{attr:<20} {baseline:<10.2f} {estimated:<10.2f} {improvement:<10.2f} {strategy}")
        estimated_improvements.append(improvement)
    
    mean_baseline = total_baseline / len(improvements)
    mean_estimated = total_estimated / len(improvements)
    overall_improvement = mean_estimated - mean_baseline
    
    print("-" * 70)
    print(f"{'平均准确率':<20} {mean_baseline:<10.2f} {mean_estimated:<10.2f} {overall_improvement:<10.2f} 总体改进")
    
    print(f"\n🎯 改进目标:")
    print(f"  当前Mean Accuracy: 74.73%")
    print(f"  目标Mean Accuracy: {mean_estimated:.2f}%")
    print(f"  预期提升: +{overall_improvement:.2f}%")
    
    if overall_improvement >= 3:
        print("  📈 预期效果: 显著改进")
    elif overall_improvement >= 1:
        print("  📈 预期效果: 中等改进")
    else:
        print("  📈 预期效果: 轻微改进")

def main():
    """主函数"""
    print("CelebA 配置对比与改进分析工具")
    print("Copyright (c) 2024")
    
    try:
        compare_attribute_groups()
        compare_loss_weights()
        compare_training_config()
        show_stage_progression()
        estimate_improvements()
        
        print(f"\n{'='*60}")
        print("✅ 配置对比分析完成")
        print("💡 建议按阶段逐步验证改进效果")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ 对比分析出错: {e}")

if __name__ == "__main__":
    main() 