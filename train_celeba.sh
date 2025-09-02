#!/bin/bash

# CelebA数据集训练脚本 - Linux服务器版本
# 弱监督解耦的跨模态属性对齐项目

echo "=========================================="
echo "CelebA 弱监督解耦训练 - Linux服务器版本"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python3 未安装"
    exit 1
fi

# 检查GPU是否可用
if python3 -c "import torch; print('GPU可用:', torch.cuda.is_available())"; then
    echo "✅ PyTorch GPU检查完成"
else
    echo "❌ 错误: PyTorch未正确安装或GPU不可用"
    exit 1
fi

# 检查数据集路径
CELEBA_PATH="/autodl-tmp"
if [ ! -d "$CELEBA_PATH" ]; then
    echo "❌ 错误: CelebA数据集路径不存在: $CELEBA_PATH"
    exit 1
fi

if [ ! -d "$CELEBA_PATH/img_align_celeba" ]; then
    echo "❌ 错误: CelebA图像目录不存在"
    exit 1
fi

if [ ! -d "$CELEBA_PATH/Anno" ]; then
    echo "❌ 错误: CelebA标注目录不存在"
    exit 1
fi

echo "✅ CelebA数据集检查通过: $CELEBA_PATH"

# 创建日志目录
mkdir -p logs

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/weak_supervised_cross_modal"

# 开始训练
echo "🚀 开始CelebA训练..."
python3 train_celeba.py 2>&1 | tee logs/train_celeba_$(date +%Y%m%d_%H%M%S).log

echo "📝 训练日志已保存到 logs/ 目录"
echo "🎯 训练完成！" 