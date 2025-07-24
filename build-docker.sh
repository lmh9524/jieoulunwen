#!/bin/bash

# 弱监督跨模态属性对齐项目 Docker 构建脚本
# 用于自动化构建和测试Docker镜像

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

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker服务未运行，请启动Docker服务"
        exit 1
    fi
    
    print_success "Docker环境检查通过"
}

# 检查NVIDIA Docker支持
check_nvidia_docker() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "检测到NVIDIA GPU，检查Docker GPU支持..."
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA Docker支持正常"
        else
            print_warning "NVIDIA Docker支持可能有问题，但仍可构建镜像"
        fi
    else
        print_warning "未检测到NVIDIA GPU，将构建CPU版本"
    fi
}

# 清理旧镜像
cleanup_old_images() {
    print_info "清理旧的镜像..."
    
    # 删除旧的镜像（如果存在）
    if docker images | grep -q "weak-supervised-cross-modal"; then
        docker rmi weak-supervised-cross-modal:latest 2>/dev/null || true
        print_info "已清理旧镜像"
    fi
    
    # 清理构建缓存
    docker builder prune -f &> /dev/null || true
}

# 构建镜像
build_image() {
    print_info "开始构建Docker镜像..."
    print_info "这可能需要几分钟时间，请耐心等待..."
    
    # 构建镜像
    if docker build -t weak-supervised-cross-modal:latest .; then
        print_success "镜像构建成功！"
        return 0
    else
        print_error "镜像构建失败！"
        return 1
    fi
}

# 显示镜像信息
show_image_info() {
    print_info "镜像信息："
    docker images weak-supervised-cross-modal:latest
    
    # 获取镜像大小
    IMAGE_SIZE=$(docker images weak-supervised-cross-modal:latest --format "{{.Size}}")
    print_info "镜像大小: $IMAGE_SIZE"
}

# 测试镜像
test_image() {
    print_info "测试镜像基本功能..."
    
    # 测试容器启动
    if docker run --rm weak-supervised-cross-modal:latest python --version; then
        print_success "Python环境测试通过"
    else
        print_error "Python环境测试失败"
        return 1
    fi
    
    # 测试PyTorch
    if docker run --rm weak-supervised-cross-modal:latest python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"; then
        print_success "PyTorch测试通过"
    else
        print_error "PyTorch测试失败"
        return 1
    fi
    
    # 测试项目模块
    if docker run --rm weak-supervised-cross-modal:latest python -c "import sys; sys.path.append('/workspace'); from weak_supervised_cross_modal.models import WeakSupervisedCrossModalAlignment; print('项目模块导入成功')"; then
        print_success "项目模块测试通过"
    else
        print_warning "项目模块测试失败，但镜像仍可使用"
    fi
}

# 导出镜像
export_image() {
    if [ "$1" = "--export" ]; then
        print_info "导出镜像为tar.gz文件..."
        docker save weak-supervised-cross-modal:latest | gzip > weak-supervised-cross-modal.tar.gz
        
        if [ -f "weak-supervised-cross-modal.tar.gz" ]; then
            EXPORT_SIZE=$(du -h weak-supervised-cross-modal.tar.gz | cut -f1)
            print_success "镜像已导出: weak-supervised-cross-modal.tar.gz (大小: $EXPORT_SIZE)"
            print_info "可以将此文件上传到AutoDL服务器"
        else
            print_error "镜像导出失败"
        fi
    fi
}

# 显示使用说明
show_usage() {
    echo ""
    print_info "镜像构建完成！以下是常用命令："
    echo ""
    echo "🚀 快速启动："
    echo "  docker run -it --gpus all weak-supervised-cross-modal:latest bash"
    echo ""
    echo "📊 训练模型："
    echo "  docker run -it --gpus all weak-supervised-cross-modal:latest train-cub --epochs 10"
    echo ""
    echo "📓 启动Jupyter："
    echo "  docker run -it --gpus all -p 8888:8888 weak-supervised-cross-modal:latest jupyter"
    echo ""
    echo "📈 启动TensorBoard："
    echo "  docker run -it --gpus all -p 6006:6006 weak-supervised-cross-modal:latest tensorboard"
    echo ""
    echo "📦 使用docker-compose："
    echo "  docker-compose up weak-supervised-cross-modal"
    echo ""
    print_info "详细使用说明请查看 Docker部署指南.md"
}

# 主函数
main() {
    echo "========================================"
    echo "弱监督跨模态属性对齐项目 Docker 构建脚本"
    echo "========================================"
    
    # 检查环境
    check_docker
    check_nvidia_docker
    
    # 清理旧镜像
    if [ "$1" = "--clean" ] || [ "$2" = "--clean" ]; then
        cleanup_old_images
    fi
    
    # 构建镜像
    if build_image; then
        show_image_info
        test_image
        export_image "$1"
        show_usage
        print_success "所有操作完成！"
    else
        print_error "构建失败，请检查错误信息"
        exit 1
    fi
}

# 显示帮助信息
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --export    构建完成后导出镜像为tar.gz文件"
    echo "  --clean     构建前清理旧镜像和缓存"
    echo "  --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 基本构建"
    echo "  $0 --export           # 构建并导出"
    echo "  $0 --clean --export   # 清理、构建并导出"
    exit 0
fi

# 执行主函数
main "$@"
