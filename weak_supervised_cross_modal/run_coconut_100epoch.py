#!/usr/bin/env python3
"""
COCONut 100轮训练启动脚本
包含参数配置和训练监控
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import yaml
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coconut_100epoch_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_environment():
    """检查环境"""
    logger.info("检查环境...")
    
    # 检查Python版本
    logger.info(f"Python版本: {sys.version}")
    
    # 检查必要的包
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 
        'matplotlib', 'tqdm', 'requests', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少必要的包: {missing_packages}")
        logger.info("请运行: pip install torch torchvision pillow numpy matplotlib tqdm requests pyyaml")
        return False
    
    return True

def check_data_directory(data_dir):
    """检查数据目录"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return False
    
    # 检查关键文件
    required_files = [
        'relabeled_coco_val.json'
    ]
    
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            logger.warning(f"数据文件不存在: {file_path}")
        else:
            logger.info(f"✓ 找到数据文件: {file_path}")
    
    return True

def run_training(config_path, **kwargs):
    """运行训练"""
    config = load_config(config_path)
    
    # 构建命令行参数
    cmd = [
        sys.executable, 
        'coconut_100_epoch_trainer.py',
        '--data_dir', str(config['data']['data_dir']),
        '--epochs', str(config['training']['epochs']),
        '--batch_size', str(config['training']['batch_size']),
        '--learning_rate', str(config['training']['learning_rate']),
        '--max_samples', str(config['data']['max_samples']),
        '--save_model', str(config['output']['model_save_path']),
        '--patience', str(config['training']['early_stopping']['patience'])
    ]
    
    # 添加额外参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("训练完成!")
            return True
        else:
            logger.error(f"训练失败，返回码: {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"运行训练时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='COCONut 100轮训练启动器')
    parser.add_argument('--config', type=str, default='configs/coconut_100epoch_config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--check_only', action='store_true', help='仅检查环境，不运行训练')
    parser.add_argument('--data_dir', type=str, help='覆盖数据目录')
    parser.add_argument('--epochs', type=int, help='覆盖训练轮数')
    parser.add_argument('--batch_size', type=int, help='覆盖批次大小')
    parser.add_argument('--learning_rate', type=float, help='覆盖学习率')
    parser.add_argument('--max_samples', type=int, help='覆盖最大样本数')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("COCONut 100轮训练启动器")
    logger.info("=" * 60)
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，请安装必要的依赖包")
        return 1
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return 1
    
    # 加载配置
    config = load_config(config_path)
    data_dir = args.data_dir or config['data']['data_dir']
    
    # 检查数据目录
    if not check_data_directory(data_dir):
        logger.error("数据目录检查失败")
        return 1
    
    if args.check_only:
        logger.info("环境检查完成，所有依赖都已就绪")
        return 0
    
    # 准备覆盖参数
    override_params = {
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_samples': args.max_samples
    }
    
    # 过滤None值
    override_params = {k: v for k, v in override_params.items() if v is not None}
    
    logger.info("开始训练...")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"训练轮数: {args.epochs or config['training']['epochs']}")
    logger.info(f"批次大小: {args.batch_size or config['training']['batch_size']}")
    logger.info(f"学习率: {args.learning_rate or config['training']['learning_rate']}")
    
    # 运行训练
    success = run_training(config_path, **override_params)
    
    if success:
        logger.info("训练成功完成!")
        return 0
    else:
        logger.error("训练失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 