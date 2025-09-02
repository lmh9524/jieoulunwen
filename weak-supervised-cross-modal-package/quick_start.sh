#!/bin/bash
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
