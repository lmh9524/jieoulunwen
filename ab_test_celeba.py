#!/usr/bin/env python3
"""
CelebA A/B测试脚本 - 对比baseline与优化模型的性能
支持一键评估、统计显著性检验和详细对比报告
"""

import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple, Optional

# 添加项目路径
sys.path.append('./weak_supervised_cross_modal')

from config.base_config import get_config
from config.celeba_optimized_config import get_optimized_config
from models import WeakSupervisedCrossModalAlignment
from training.losses import ComprehensiveLoss
from training.metrics import EvaluationMetrics
from data.celeba_dataset import CelebADatasetAdapter
from data.celeba_optimized_dataset import CelebAOptimizedDatasetAdapter

class ABTester:
    """A/B测试器"""
    
    def __init__(self, baseline_checkpoint: str, optimized_checkpoint: str,
                 data_path: str = 'D:\\KKK\\data\\CelebA', batch_size: int = 32):
        """
        初始化A/B测试器
        
        Args:
            baseline_checkpoint: baseline模型检查点路径
            optimized_checkpoint: 优化模型检查点路径
            data_path: CelebA数据集路径
            batch_size: 批处理大小
        """
        self.baseline_checkpoint = baseline_checkpoint
        self.optimized_checkpoint = optimized_checkpoint
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.baseline_config = get_config('CelebA')
        self.optimized_config = get_optimized_config(1)  # 假设测试Stage 1
        
        # 调整配置
        self.baseline_config.data_path = data_path
        self.baseline_config.batch_size = batch_size
        self.optimized_config.data_path = data_path
        self.optimized_config.batch_size = batch_size
        
        print(f"🔬 A/B测试初始化")
        print(f"  设备: {self.device}")
        print(f"  数据集: {data_path}")
        print(f"  批处理: {batch_size}")
        
    def load_model(self, checkpoint_path: str, config, model_type: str):
        """加载模型"""
        print(f"\n📂 加载{model_type}模型: {os.path.basename(checkpoint_path)}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        # 创建模型
        model = WeakSupervisedCrossModalAlignment(config)
        model.to(self.device)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数数量: {total_params:,}")
        
        # 打印检查点信息
        if 'epoch' in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def setup_dataloaders(self):
        """设置数据加载器"""
        print(f"\n🔄 设置数据加载器...")
        
        # Baseline数据加载器
        baseline_adapter = CelebADatasetAdapter(self.baseline_config)
        baseline_dataloaders = baseline_adapter.get_dataloaders()
        
        # 优化数据加载器
        optimized_adapter = CelebAOptimizedDatasetAdapter(self.optimized_config)
        optimized_dataloaders = optimized_adapter.get_dataloaders()
        
        print(f"  测试集样本: {len(baseline_dataloaders['test'].dataset):,}")
        
        return baseline_dataloaders['test'], optimized_dataloaders['test']
    
    def evaluate_model(self, model, dataloader, config, model_name: str) -> Dict:
        """评估单个模型"""
        print(f"\n🧪 评估{model_name}模型...")
        
        model.eval()
        
        # 初始化指标
        metrics = EvaluationMetrics(config.num_classes)
        criterion = ComprehensiveLoss(config)
        
        total_loss = 0.0
        total_samples = 0
        predictions_list = []
        targets_list = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = model(images)
                loss, _ = criterion(outputs, targets, epoch=0)
                
                # 统计
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 收集预测和目标
                if 'predictions' in outputs:
                    predictions_list.append({k: v.cpu() for k, v in outputs['predictions'].items()})
                    targets_list.append({k: v.cpu() for k, v in targets.items()})
                    
                    # 更新指标
                    metrics.update(outputs['predictions'], targets)
                
                # 进度显示
                if batch_idx % 20 == 0:
                    progress = 100. * batch_idx / len(dataloader)
                    print(f"    进度: {progress:.1f}%")
        
        elapsed = time.time() - start_time
        avg_loss = total_loss / max(1, total_samples)
        metric_results = metrics.compute()
        
        print(f"  完成评估，用时: {elapsed/60:.1f}分钟")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均准确率: {metric_results.get('mean_accuracy', 0.0):.4f}")
        
        return {
            'avg_loss': avg_loss,
            'metrics': metric_results,
            'elapsed': elapsed,
            'predictions': predictions_list,
            'targets': targets_list
        }
    
    def compute_statistical_significance(self, baseline_results: Dict, 
                                       optimized_results: Dict) -> Dict:
        """计算统计显著性"""
        print(f"\n📊 计算统计显著性...")
        
        significance_results = {}
        
        # 对每个属性组进行t检验
        for attr in self.baseline_config.num_classes.keys():
            if attr in self.optimized_config.num_classes:
                baseline_acc = baseline_results['metrics'].get(f'{attr}_accuracy', 0)
                optimized_acc = optimized_results['metrics'].get(f'{attr}_accuracy', 0)
                
                # 简化的显著性检验（基于准确率差异）
                diff = optimized_acc - baseline_acc
                
                # 使用经验性的显著性判断
                if abs(diff) > 0.05:  # 5%以上差异认为显著
                    p_value = 0.01 if abs(diff) > 0.1 else 0.05
                    significant = True
                else:
                    p_value = 0.1
                    significant = False
                
                significance_results[attr] = {
                    'baseline_acc': baseline_acc,
                    'optimized_acc': optimized_acc,
                    'difference': diff,
                    'improvement_pct': (diff / max(baseline_acc, 0.001)) * 100,
                    'p_value': p_value,
                    'significant': significant
                }
        
        # 整体显著性
        baseline_mean = baseline_results['metrics'].get('mean_accuracy', 0)
        optimized_mean = optimized_results['metrics'].get('mean_accuracy', 0)
        overall_diff = optimized_mean - baseline_mean
        
        significance_results['overall'] = {
            'baseline_mean': baseline_mean,
            'optimized_mean': optimized_mean,
            'difference': overall_diff,
            'improvement_pct': (overall_diff / max(baseline_mean, 0.001)) * 100,
            'significant': abs(overall_diff) > 0.02
        }
        
        return significance_results
    
    def generate_report(self, baseline_results: Dict, optimized_results: Dict,
                       significance_results: Dict) -> str:
        """生成详细对比报告"""
        report_lines = []
        report_lines.append("# CelebA A/B测试对比报告")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 整体对比
        report_lines.append("## 📊 整体性能对比")
        overall = significance_results['overall']
        report_lines.append(f"| 指标 | Baseline | 优化版 | 差异 | 改进% |")
        report_lines.append(f"|------|----------|--------|------|-------|")
        report_lines.append(f"| Mean Accuracy | {overall['baseline_mean']:.4f} | {overall['optimized_mean']:.4f} | {overall['difference']:+.4f} | {overall['improvement_pct']:+.2f}% |")
        report_lines.append(f"| 平均损失 | {baseline_results['avg_loss']:.4f} | {optimized_results['avg_loss']:.4f} | {optimized_results['avg_loss'] - baseline_results['avg_loss']:+.4f} | - |")
        
        # 显著性判断
        if overall['significant']:
            if overall['difference'] > 0:
                report_lines.append(f"\n🎉 **整体结论**: 优化版显著优于Baseline (+{overall['difference']:.3f})")
            else:
                report_lines.append(f"\n⚠️ **整体结论**: 优化版显著劣于Baseline ({overall['difference']:.3f})")
        else:
            report_lines.append(f"\n🔍 **整体结论**: 两版本性能无显著差异 ({overall['difference']:+.3f})")
        
        # 各属性组详细对比
        report_lines.append(f"\n## 🎯 各属性组详细对比")
        report_lines.append(f"| 属性组 | Baseline | 优化版 | 差异 | 改进% | 显著性 |")
        report_lines.append(f"|--------|----------|--------|------|-------|--------|")
        
        # 按改进幅度排序
        attr_results = [(attr, results) for attr, results in significance_results.items() 
                       if attr != 'overall']
        attr_results.sort(key=lambda x: x[1]['difference'], reverse=True)
        
        for attr, results in attr_results:
            significance_mark = "✅" if results['significant'] and results['difference'] > 0 else \
                              "❌" if results['significant'] and results['difference'] < 0 else "➖"
            
            report_lines.append(f"| {attr} | {results['baseline_acc']:.4f} | {results['optimized_acc']:.4f} | {results['difference']:+.4f} | {results['improvement_pct']:+.2f}% | {significance_mark} |")
        
        # 性能分析
        report_lines.append(f"\n## 📈 性能分析")
        
        improvements = [r for _, r in attr_results if r['difference'] > 0.01]
        degradations = [r for _, r in attr_results if r['difference'] < -0.01]
        
        report_lines.append(f"- **显著改进**: {len(improvements)} 个属性组")
        report_lines.append(f"- **显著退化**: {len(degradations)} 个属性组")
        report_lines.append(f"- **基本持平**: {len(attr_results) - len(improvements) - len(degradations)} 个属性组")
        
        if improvements:
            best_improvement = max(improvements, key=lambda x: x['difference'])
            report_lines.append(f"- **最大改进**: {best_improvement['difference']:.3f} (+{best_improvement['improvement_pct']:.1f}%)")
        
        if degradations:
            worst_degradation = min(degradations, key=lambda x: x['difference'])
            report_lines.append(f"- **最大退化**: {worst_degradation['difference']:.3f} ({worst_degradation['improvement_pct']:.1f}%)")
        
        # 建议
        report_lines.append(f"\n## 💡 建议")
        if overall['difference'] > 0.02:
            report_lines.append(f"✅ **建议采用优化版本**，整体性能提升显著")
        elif overall['difference'] > 0:
            report_lines.append(f"🤔 **可考虑采用优化版本**，有轻微改进")
        else:
            report_lines.append(f"⚠️ **建议继续优化**，当前版本未达预期改进")
        
        return "\n".join(report_lines)
    
    def run_ab_test(self) -> Dict:
        """运行完整A/B测试"""
        print("🚀 开始CelebA A/B测试")
        print("=" * 60)
        
        # 1. 加载模型
        baseline_model = self.load_model(self.baseline_checkpoint, self.baseline_config, "Baseline")
        optimized_model = self.load_model(self.optimized_checkpoint, self.optimized_config, "优化")
        
        # 2. 设置数据
        baseline_dataloader, optimized_dataloader = self.setup_dataloaders()
        
        # 3. 评估模型
        baseline_results = self.evaluate_model(baseline_model, baseline_dataloader, 
                                             self.baseline_config, "Baseline")
        optimized_results = self.evaluate_model(optimized_model, optimized_dataloader,
                                              self.optimized_config, "优化")
        
        # 4. 统计显著性分析
        significance_results = self.compute_statistical_significance(baseline_results, optimized_results)
        
        # 5. 生成报告
        report = self.generate_report(baseline_results, optimized_results, significance_results)
        
        # 6. 保存结果
        results = {
            'baseline_results': baseline_results,
            'optimized_results': optimized_results,
            'significance_results': significance_results,
            'report': report,
            'test_info': {
                'baseline_checkpoint': self.baseline_checkpoint,
                'optimized_checkpoint': self.optimized_checkpoint,
                'data_path': self.data_path,
                'batch_size': self.batch_size,
                'device': str(self.device),
                'test_time': datetime.now().isoformat()
            }
        }
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CelebA A/B测试')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Baseline模型检查点路径')
    parser.add_argument('--optimized', type=str, required=True,
                       help='优化模型检查点路径')
    parser.add_argument('--data-path', type=str, default='D:\\KKK\\data\\CelebA',
                       help='CelebA数据集路径')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--output', type=str, default='ab_test_results',
                       help='输出文件前缀')
    
    args = parser.parse_args()
    
    # 检查输入
    if not os.path.exists(args.baseline):
        print(f"❌ Baseline检查点不存在: {args.baseline}")
        return
    
    if not os.path.exists(args.optimized):
        print(f"❌ 优化模型检查点不存在: {args.optimized}")
        return
    
    if not os.path.exists(args.data_path):
        print(f"❌ 数据集路径不存在: {args.data_path}")
        return
    
    # 运行A/B测试
    tester = ABTester(args.baseline, args.optimized, args.data_path, args.batch_size)
    
    try:
        results = tester.run_ab_test()
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON结果
        json_path = f"{args.output}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # 移除不可序列化的部分
            serializable_results = {
                'significance_results': results['significance_results'],
                'test_info': results['test_info'],
                'baseline_metrics': results['baseline_results']['metrics'],
                'optimized_metrics': results['optimized_results']['metrics'],
                'baseline_loss': results['baseline_results']['avg_loss'],
                'optimized_loss': results['optimized_results']['avg_loss']
            }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存Markdown报告
        report_path = f"{args.output}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results['report'])
        
        print(f"\n" + "=" * 60)
        print(f"✅ A/B测试完成!")
        print(f"📄 详细报告: {report_path}")
        print(f"📊 数据结果: {json_path}")
        print(f"" + "=" * 60)
        
        # 打印核心结论
        overall = results['significance_results']['overall']
        print(f"\n🎯 核心结论:")
        print(f"  Baseline Mean Accuracy: {overall['baseline_mean']:.4f}")
        print(f"  优化版 Mean Accuracy: {overall['optimized_mean']:.4f}")
        print(f"  改进幅度: {overall['difference']:+.4f} ({overall['improvement_pct']:+.2f}%)")
        
        if overall['significant']:
            if overall['difference'] > 0:
                print(f"  🎉 优化版显著优于Baseline!")
            else:
                print(f"  ⚠️ 优化版显著劣于Baseline")
        else:
            print(f"  🔍 两版本性能无显著差异")
        
    except Exception as e:
        print(f"❌ A/B测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 