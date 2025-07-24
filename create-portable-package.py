#!/usr/bin/env python3
"""
创建便携式项目包，用于在AutoDL服务器上部署
当Docker网络有问题时的替代方案
"""

import os
import shutil
import zipfile
import json
from pathlib import Path

def create_portable_package():
    """创建便携式项目包"""
    
    print("🚀 开始创建便携式项目包...")
    
    # 创建打包目录
    package_dir = Path("weak-supervised-cross-modal-package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # 复制项目代码
    print("📁 复制项目代码...")
    shutil.copytree("weak_supervised_cross_modal", package_dir / "weak_supervised_cross_modal")
    
    # 复制重要文件
    important_files = [
        "*.pth",  # 模型文件
        "*.json", # 结果文件
        "*.md",   # 文档文件
    ]
    
    for pattern in important_files:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                shutil.copy2(file_path, package_dir)
                print(f"✅ 复制文件: {file_path}")
    
    # 复制目录
    important_dirs = ["checkpoints", "checkpoints_coco", "configs"]
    for dir_name in important_dirs:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, package_dir / dir_name)
            print(f"✅ 复制目录: {dir_name}")
    
    # 复制数据集（选择性）
    data_dir = Path("data")
    if data_dir.exists():
        target_data_dir = package_dir / "data"
        target_data_dir.mkdir()
        
        # 只复制重要的数据文件，不复制大型图片
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                target_subdir = target_data_dir / subdir.name
                target_subdir.mkdir()
                
                # 复制配置和标注文件
                for file_path in subdir.rglob("*.json"):
                    rel_path = file_path.relative_to(subdir)
                    target_file = target_subdir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_file)
                
                for file_path in subdir.rglob("*.txt"):
                    rel_path = file_path.relative_to(subdir)
                    target_file = target_subdir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_file)
    
    # 创建安装脚本
    install_script = """#!/bin/bash
# AutoDL服务器安装脚本

echo "🚀 开始安装弱监督跨模态属性对齐项目..."

# 检查Python环境
python --version
pip --version

# 安装PyTorch (CUDA版本)
echo "📦 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
echo "📦 安装项目依赖..."
pip install -r weak_supervised_cross_modal/requirements.txt

# 安装额外工具
pip install jupyter jupyterlab tensorboard wandb

echo "✅ 安装完成！"
echo ""
echo "🎯 使用方法："
echo "cd weak_supervised_cross_modal"
echo "python main.py --dataset cub --mode train --epochs 10"
echo ""
echo "📓 启动Jupyter："
echo "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
"""
    
    with open(package_dir / "install.sh", "w", encoding="utf-8") as f:
        f.write(install_script)
    
    # 创建快速启动脚本
    quick_start = """#!/bin/bash
# 快速启动脚本

echo "🎯 弱监督跨模态属性对齐项目"
echo "选择运行模式："
echo "1. 训练CUB模型"
echo "2. 训练COCO属性模型" 
echo "3. 训练COCONut模型"
echo "4. 运行推理"
echo "5. 启动Jupyter Lab"
echo "6. 启动TensorBoard"

read -p "请选择 (1-6): " choice

case $choice in
    1)
        echo "🚀 开始训练CUB模型..."
        cd weak_supervised_cross_modal
        python main.py --dataset cub --mode train --epochs 50
        ;;
    2)
        echo "🚀 开始训练COCO属性模型..."
        cd weak_supervised_cross_modal
        python train_coco_attributes.py --epochs 40
        ;;
    3)
        echo "🚀 开始训练COCONut模型..."
        cd weak_supervised_cross_modal
        python run_coconut_100epoch.py --epochs 100
        ;;
    4)
        echo "🔍 运行推理..."
        cd weak_supervised_cross_modal
        python inference.py
        ;;
    5)
        echo "📓 启动Jupyter Lab..."
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    6)
        echo "📈 启动TensorBoard..."
        tensorboard --logdir=logs --host=0.0.0.0 --port=6006
        ;;
    *)
        echo "❌ 无效选择"
        ;;
esac
"""
    
    with open(package_dir / "quick_start.sh", "w", encoding="utf-8") as f:
        f.write(quick_start)
    
    # 创建README
    readme_content = """# 弱监督跨模态属性对齐项目 - 便携式包

## 🚀 快速开始

### 1. 安装依赖
```bash
chmod +x install.sh
./install.sh
```

### 2. 快速启动
```bash
chmod +x quick_start.sh
./quick_start.sh
```

### 3. 手动运行
```bash
cd weak_supervised_cross_modal

# 训练CUB模型
python main.py --dataset cub --mode train --epochs 50

# 训练COCO属性模型
python train_coco_attributes.py --epochs 40

# 运行推理
python inference.py
```

## 📁 目录结构
- `weak_supervised_cross_modal/` - 主要代码
- `checkpoints/` - 模型检查点
- `data/` - 数据集配置文件
- `*.pth` - 预训练模型
- `install.sh` - 安装脚本
- `quick_start.sh` - 快速启动脚本

## 🔧 AutoDL使用
1. 上传此包到AutoDL服务器
2. 解压: `unzip weak-supervised-cross-modal-package.zip`
3. 运行安装脚本: `./install.sh`
4. 开始使用: `./quick_start.sh`

## 📞 技术支持
如有问题，请查看项目文档或检查日志输出。
"""
    
    with open(package_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # 创建项目信息文件
    project_info = {
        "name": "弱监督跨模态属性对齐",
        "version": "1.0.0",
        "description": "Weak Supervised Cross-Modal Attribute Alignment",
        "created": "2025-01-09",
        "python_version": "3.8+",
        "pytorch_version": "2.0+",
        "cuda_version": "11.8",
        "main_files": [
            "weak_supervised_cross_modal/main.py",
            "weak_supervised_cross_modal/train_coco_attributes.py",
            "weak_supervised_cross_modal/run_coconut_100epoch.py",
            "weak_supervised_cross_modal/inference.py"
        ]
    }
    
    with open(package_dir / "project_info.json", "w", encoding="utf-8") as f:
        json.dump(project_info, f, indent=2, ensure_ascii=False)
    
    # 创建ZIP包
    print("📦 创建ZIP包...")
    zip_path = "weak-supervised-cross-modal-package.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(package_dir.parent)
                zipf.write(file_path, arc_path)
    
    # 获取包大小
    package_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    
    print(f"✅ 便携式包创建完成！")
    print(f"📦 包文件: {zip_path}")
    print(f"📏 包大小: {package_size:.1f} MB")
    print(f"📁 包目录: {package_dir}")
    
    print("\n🎯 使用方法:")
    print("1. 将 weak-supervised-cross-modal-package.zip 上传到AutoDL服务器")
    print("2. 解压: unzip weak-supervised-cross-modal-package.zip")
    print("3. 进入目录: cd weak-supervised-cross-modal-package")
    print("4. 安装依赖: ./install.sh")
    print("5. 开始使用: ./quick_start.sh")
    
    return zip_path

if __name__ == "__main__":
    create_portable_package()
