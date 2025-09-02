#!/bin/bash
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
