#!/usr/bin/env python3
"""
CelebA训练监控脚本
实时监控CelebA训练进度和性能指标
"""

import os
import time
import json
import glob
from pathlib import Path
import psutil

class CelebATrainingMonitor:
    def __init__(self):
        self.experiments_dir = Path("./experiments")
        self.current_experiment = None
        
        print("CelebA训练监控器启动")
        print(f"监控目录: {self.experiments_dir}")
    
    def find_current_experiment(self):
        """查找当前正在进行的实验"""
        if not self.experiments_dir.exists():
            return None
        
        # 查找最新的CelebA实验目录
        celeba_experiments = list(self.experiments_dir.glob("celeba_training_*"))
        if not celeba_experiments:
            return None
        
        # 按修改时间排序，获取最新的
        latest_experiment = max(celeba_experiments, key=lambda x: x.stat().st_mtime)
        return latest_experiment
    
    def check_training_process(self):
        """检查是否有训练进程在运行"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('train_celeba.py' in str(arg) for arg in cmdline):
                    return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, None
    
    def read_training_history(self, experiment_dir):
        """读取训练历史"""
        history_file = experiment_dir / "training_history.json"
        if not history_file.exists():
            return None
        
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def read_config(self, experiment_dir):
        """读取实验配置"""
        config_file = experiment_dir / "config.json"
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def count_checkpoints(self, experiment_dir):
        """统计检查点数量"""
        checkpoints_dir = experiment_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return 0, False
        
        checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
        best_model_exists = (checkpoints_dir / "best_model.pth").exists()
        
        return len(checkpoints), best_model_exists
    
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name()
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                
                return {
                    'available': True,
                    'name': device_name,
                    'memory_total': memory_total,
                    'memory_used': memory_allocated,
                    'memory_cached': memory_cached
                }
        except:
            pass
        
        return {'available': False}
    
    def display_status(self, experiment_dir, is_running, pid):
        """显示当前状态"""
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        
        print("=" * 70)
        print("CelebA 弱监督解耦训练监控")
        print("=" * 70)
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"实验目录: {experiment_dir.name}")
        
        if is_running:
            print(f"状态: 🟢 训练进行中 (PID: {pid})")
        else:
            print("状态: 🔴 训练进程未运行")
        
        # 读取配置
        config = self.read_config(experiment_dir)
        if config:
            print(f"\n📋 训练配置:")
            print(f"  数据集: {config.get('dataset_name', 'CelebA')}")
            print(f"  总轮数: {config.get('num_epochs', 'N/A')}")
            print(f"  批大小: {config.get('batch_size', 'N/A')}")
            print(f"  学习率: {config.get('learning_rate', 'N/A')}")
            print(f"  图像尺寸: {config.get('image_size', 'N/A')}")
        
        # 读取训练历史
        history = self.read_training_history(experiment_dir)
        if history:
            epochs_completed = len(history['train_loss'])
            total_epochs = config.get('num_epochs', epochs_completed)
            
            print(f"\n📊 训练进度:")
            print(f"  已完成轮数: {epochs_completed}/{total_epochs}")
            
            if epochs_completed > 0:
                latest_train_loss = history['train_loss'][-1]
                latest_val_loss = history['val_loss'][-1]
                latest_train_acc = history['train_acc'][-1]
                latest_val_acc = history['val_acc'][-1]
                
                print(f"  最新训练损失: {latest_train_loss:.4f}")
                print(f"  最新验证损失: {latest_val_loss:.4f}")
                print(f"  最新训练准确率: {latest_train_acc:.4f}")
                print(f"  最新验证准确率: {latest_val_acc:.4f}")
                
                # 显示最佳性能
                best_val_loss = min(history['val_loss'])
                best_val_acc = max(history['val_acc'])
                
                print(f"  最佳验证损失: {best_val_loss:.4f}")
                print(f"  最佳验证准确率: {best_val_acc:.4f}")
                
                # 进度条
                progress = (epochs_completed / total_epochs) * 100 if total_epochs > 0 else 0
                bar_length = 40
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"  进度: [{bar}] {progress:.1f}%")
        
        # 检查点信息
        checkpoint_count, best_model_exists = self.count_checkpoints(experiment_dir)
        print(f"\n💾 模型检查点:")
        print(f"  检查点数量: {checkpoint_count}")
        print(f"  最佳模型: {'✅ 已保存' if best_model_exists else '❌ 未保存'}")
        
        # GPU信息
        gpu_info = self.get_gpu_info()
        if gpu_info['available']:
            print(f"\n🖥️  GPU状态:")
            print(f"  设备: {gpu_info['name']}")
            print(f"  总内存: {gpu_info['memory_total']:.1f} GB")
            print(f"  已使用: {gpu_info['memory_used']:.1f} GB")
            print(f"  缓存: {gpu_info['memory_cached']:.1f} GB")
        else:
            print(f"\n🖥️  GPU状态: 不可用")
        
        print(f"\n💡 提示:")
        print("- 按 Ctrl+C 退出监控")
        print("- 训练进程在后台运行，关闭监控不影响训练")
        
        # 检查是否有训练报告
        report_file = experiment_dir / "training_report.json"
        if report_file.exists():
            print("- 发现训练完成报告，训练可能已完成")
        
        print("=" * 70)
    
    def monitor(self, refresh_interval=10):
        """开始监控"""
        print(f"开始监控，每{refresh_interval}秒刷新一次...")
        
        try:
            while True:
                # 查找当前实验
                current_exp = self.find_current_experiment()
                
                if not current_exp:
                    print("未找到CelebA训练实验")
                    time.sleep(refresh_interval)
                    continue
                
                # 检查训练进程状态
                is_running, pid = self.check_training_process()
                
                # 显示状态
                self.display_status(current_exp, is_running, pid)
                
                # 检查是否完成
                report_file = current_exp / "training_report.json"
                if report_file.exists() and not is_running:
                    print("\n🎉 训练已完成！")
                    
                    # 显示最终报告
                    try:
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                        
                        print(f"\n📋 最终训练报告:")
                        print(f"  实验名称: {report.get('experiment_name', 'N/A')}")
                        print(f"  训练时间: {report.get('total_time_hours', 0):.2f} 小时")
                        print(f"  最佳验证损失: {report.get('best_val_loss', 0):.4f}")
                        print(f"  最终训练准确率: {report.get('final_train_acc', 0):.4f}")
                        print(f"  最终验证准确率: {report.get('final_val_acc', 0):.4f}")
                        print(f"  模型参数量: {report.get('model_parameters', 0):,}")
                    except:
                        print("无法读取训练报告详情")
                    
                    break
                
                # 等待下次刷新
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            if is_running:
                print("训练进程仍在后台运行...")

def main():
    """主函数"""
    monitor = CelebATrainingMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main() 