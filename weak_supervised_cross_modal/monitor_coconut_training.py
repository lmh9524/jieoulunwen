#!/usr/bin/env python3
"""
监控COCONut训练进度
"""

import time
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def monitor_training():
    """监控训练进度"""
    print("🥥 COCONut 100轮训练监控")
    print("=" * 60)
    print("监控完整数据集训练进度...")
    print("数据集：5000张真实图像")
    print("=" * 60)
    
    # 查找最新的训练日志或模型文件
    log_patterns = [
        "coconut_*.log",
        "real_training_*.png",
        "coconut_real_100epoch*.pth"
    ]
    
    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] 检查训练状态...")
        
        # 检查是否有训练文件生成
        training_files = []
        for pattern in log_patterns:
            training_files.extend(glob.glob(pattern))
        
        if training_files:
            print(f"找到训练文件: {len(training_files)} 个")
            for file in training_files:
                stat = os.stat(file)
                print(f"  - {file} (大小: {stat.st_size} bytes)")
        else:
            print("暂未找到训练输出文件")
        
        # 检查GPU使用情况
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU状态: {torch.cuda.get_device_name(0)}")
                print(f"GPU内存: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB / {torch.cuda.memory_reserved(0)/1024**3:.1f}GB")
        except:
            pass
        
        # 检查进程
        try:
            import psutil
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'] and 'coconut' in ' '.join(proc.info['cmdline']):
                        python_processes.append(proc.info)
                except:
                    pass
            
            if python_processes:
                print(f"发现训练进程: {len(python_processes)} 个")
                for proc in python_processes:
                    print(f"  - PID: {proc['pid']}")
            else:
                print("未发现训练进程")
                
        except ImportError:
            print("无法检查进程状态 (需要安装psutil)")
        
        print("-" * 40)
        time.sleep(30)  # 每30秒检查一次

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n监控已停止") 