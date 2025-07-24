#!/bin/bash

# 弱监督跨模态属性对齐项目 Docker 启动脚本
# 支持多种运行模式：训练、推理、Jupyter、交互式

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "弱监督跨模态属性对齐项目 Docker 容器"
    echo ""
    echo "使用方法:"
    echo "  docker run -it --gpus all your-image:tag [COMMAND] [OPTIONS]"
    echo ""
    echo "可用命令:"
    echo "  train-cub          - 训练CUB数据集模型"
    echo "  train-coco         - 训练COCO属性模型"
    echo "  train-coconut      - 训练COCONut模型"
    echo "  inference          - 运行推理"
    echo "  jupyter            - 启动Jupyter Lab"
    echo "  tensorboard        - 启动TensorBoard"
    echo "  bash               - 进入交互式shell"
    echo "  help               - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  docker run -it --gpus all your-image:tag train-cub --epochs 10"
    echo "  docker run -it --gpus all -p 8888:8888 your-image:tag jupyter"
    echo "  docker run -it --gpus all your-image:tag bash"
}

# 检查GPU可用性
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "检查GPU状态..."
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
        print_success "GPU可用，设备: $CUDA_VISIBLE_DEVICES"
    else
        print_warning "未检测到GPU，将使用CPU模式"
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# 设置环境
setup_environment() {
    print_info "设置环境变量..."
    export PYTHONPATH=/workspace:$PYTHONPATH
    export TORCH_HOME=/workspace/.torch
    
    # 创建必要目录
    mkdir -p /workspace/logs /workspace/results /workspace/checkpoints
    
    cd /workspace
    print_success "环境设置完成"
}

# 训练CUB模型
train_cub() {
    print_info "开始训练CUB模型..."
    cd /workspace/weak_supervised_cross_modal
    python main.py --dataset cub --mode train "$@"
}

# 训练COCO属性模型
train_coco() {
    print_info "开始训练COCO属性模型..."
    cd /workspace/weak_supervised_cross_modal
    python train_coco_attributes.py "$@"
}

# 训练COCONut模型
train_coconut() {
    print_info "开始训练COCONut模型..."
    cd /workspace/weak_supervised_cross_modal
    python run_coconut_100epoch.py "$@"
}

# 运行推理
run_inference() {
    print_info "运行推理..."
    cd /workspace/weak_supervised_cross_modal
    python inference.py "$@"
}

# 启动Jupyter Lab
start_jupyter() {
    print_info "启动Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --notebook-dir=/workspace \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.allow_remote_access=True
}

# 启动TensorBoard
start_tensorboard() {
    print_info "启动TensorBoard..."
    tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
}

# 主函数
main() {
    print_info "弱监督跨模态属性对齐项目 Docker 容器启动"
    
    # 设置环境
    setup_environment
    check_gpu
    
    # 根据参数执行不同命令
    case "${1:-bash}" in
        "train-cub")
            shift
            train_cub "$@"
            ;;
        "train-coco")
            shift
            train_coco "$@"
            ;;
        "train-coconut")
            shift
            train_coconut "$@"
            ;;
        "inference")
            shift
            run_inference "$@"
            ;;
        "jupyter")
            start_jupyter
            ;;
        "tensorboard")
            start_tensorboard
            ;;
        "help")
            show_help
            ;;
        "bash"|"sh")
            print_info "进入交互式shell..."
            exec bash
            ;;
        *)
            print_info "执行自定义命令: $@"
            exec "$@"
            ;;
    esac
}

# 执行主函数
main "$@"
