# 弱监督跨模态属性对齐项目 Docker 部署指南

## 🚀 快速开始

### 1. 构建Docker镜像

```bash
# 在项目根目录下构建镜像
docker build -t weak-supervised-cross-modal:latest .

# 或者使用docker-compose构建
docker-compose build
```

### 2. 运行容器

#### 方式一：直接使用Docker命令

```bash
# 交互式运行（推荐用于开发和调试）
docker run -it --gpus all \
  -p 8888:8888 -p 6006:6006 \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/logs:/workspace/logs \
  weak-supervised-cross-modal:latest bash

# 后台运行训练
docker run -d --gpus all \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/logs:/workspace/logs \
  weak-supervised-cross-modal:latest train-cub --epochs 50
```

#### 方式二：使用docker-compose（推荐）

```bash
# 启动主服务（交互式）
docker-compose up weak-supervised-cross-modal

# 启动Jupyter Lab
docker-compose --profile jupyter up jupyter

# 启动TensorBoard
docker-compose --profile tensorboard up tensorboard

# 后台训练
docker-compose --profile training up training
```

## 📋 可用命令

### 训练命令

```bash
# CUB数据集训练
docker run -it --gpus all weak-supervised-cross-modal:latest train-cub --epochs 50 --batch-size 32

# COCO属性训练
docker run -it --gpus all weak-supervised-cross-modal:latest train-coco --epochs 40

# COCONut训练
docker run -it --gpus all weak-supervised-cross-modal:latest train-coconut --epochs 100
```

### 推理命令

```bash
# 运行推理
docker run -it --gpus all \
  -v $(pwd)/results:/workspace/results \
  weak-supervised-cross-modal:latest inference --model /workspace/best_model.pth
```

### 开发工具

```bash
# 启动Jupyter Lab
docker run -it --gpus all -p 8888:8888 weak-supervised-cross-modal:latest jupyter

# 启动TensorBoard
docker run -it --gpus all -p 6006:6006 weak-supervised-cross-modal:latest tensorboard

# 进入交互式shell
docker run -it --gpus all weak-supervised-cross-modal:latest bash
```

## 🔧 AutoDL服务器部署

### 1. 上传镜像到AutoDL

```bash
# 方法一：保存镜像为tar文件
docker save weak-supervised-cross-modal:latest | gzip > weak-supervised-cross-modal.tar.gz

# 上传到AutoDL服务器后加载
gunzip -c weak-supervised-cross-modal.tar.gz | docker load
```

### 2. 在AutoDL上运行

```bash
# 检查GPU
nvidia-smi

# 运行容器
docker run -it --gpus all \
  -p 8888:8888 -p 6006:6006 \
  -v /root/autodl-tmp:/workspace/results \
  weak-supervised-cross-modal:latest bash
```

### 3. AutoDL优化配置

```bash
# 设置共享内存大小（避免DataLoader问题）
docker run -it --gpus all --shm-size=8g \
  weak-supervised-cross-modal:latest train-cub

# 使用AutoDL的数据盘
docker run -it --gpus all \
  -v /root/autodl-tmp:/workspace/results \
  -v /root/autodl-nas:/workspace/data \
  weak-supervised-cross-modal:latest
```

## 📁 目录结构

容器内的目录结构：

```
/workspace/
├── weak_supervised_cross_modal/    # 主要代码
├── data/                          # 数据集
├── checkpoints/                   # 模型检查点
├── results/                       # 训练结果
├── logs/                         # 日志文件
├── docs/                         # 文档
└── configs/                      # 配置文件
```

## 🔍 故障排除

### 常见问题

#### 1. GPU不可用
```bash
# 检查NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 如果失败，安装nvidia-container-toolkit
```

#### 2. 内存不足
```bash
# 减少batch size
docker run -it --gpus all weak-supervised-cross-modal:latest train-cub --batch-size 16

# 增加共享内存
docker run -it --gpus all --shm-size=8g weak-supervised-cross-modal:latest
```

#### 3. 端口冲突
```bash
# 使用不同端口
docker run -it --gpus all -p 8889:8888 weak-supervised-cross-modal:latest jupyter
```

### 调试技巧

```bash
# 查看容器日志
docker logs container_name

# 进入运行中的容器
docker exec -it container_name bash

# 监控GPU使用
docker exec -it container_name nvidia-smi

# 查看容器资源使用
docker stats container_name
```

## 📊 性能优化

### 1. 镜像优化
- 使用多阶段构建减小镜像大小
- 清理不必要的缓存和临时文件
- 使用.dockerignore排除无关文件

### 2. 运行时优化
- 合理设置batch_size
- 使用混合精度训练
- 启用CUDA优化

### 3. 存储优化
- 使用卷挂载持久化数据
- 定期清理日志和临时文件
- 使用高速存储存放数据集

## 🔐 安全注意事项

- 不要在生产环境中使用空密码的Jupyter
- 限制容器的网络访问
- 定期更新基础镜像
- 使用非root用户运行（如需要）

## 📞 技术支持

如果遇到问题，请检查：
1. Docker和NVIDIA Docker是否正确安装
2. GPU驱动是否兼容
3. 容器日志中的错误信息
4. 系统资源是否充足

## 🛠️ 构建脚本

为了简化构建过程，可以使用以下脚本：

```bash
#!/bin/bash
# build-docker.sh

echo "开始构建弱监督跨模态属性对齐项目Docker镜像..."

# 构建镜像
docker build -t weak-supervised-cross-modal:latest .

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✅ 镜像构建成功！"
    echo "镜像大小："
    docker images weak-supervised-cross-modal:latest

    echo ""
    echo "快速测试命令："
    echo "docker run -it --gpus all weak-supervised-cross-modal:latest bash"
else
    echo "❌ 镜像构建失败！"
    exit 1
fi
```

## 📦 镜像导出和导入

```bash
# 导出镜像（用于上传到AutoDL）
docker save weak-supervised-cross-modal:latest | gzip > weak-supervised-cross-modal.tar.gz

# 在AutoDL服务器上导入
gunzip -c weak-supervised-cross-modal.tar.gz | docker load

# 验证导入成功
docker images | grep weak-supervised-cross-modal
```

---

**提示**: 首次运行建议使用交互式模式进行测试，确认环境正常后再进行批量训练。
