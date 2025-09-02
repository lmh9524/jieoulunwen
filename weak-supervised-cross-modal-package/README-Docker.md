# 弱监督跨模态属性对齐项目 - Docker版本

## 🎯 项目概述

本项目已完全Docker化，可以在AutoDL服务器上直接运行，避免长时间上传和环境配置问题。

## 📦 包含内容

- **完整的训练环境**: PyTorch 2.1.0 + CUDA 11.8
- **所有项目代码**: 弱监督跨模态属性对齐算法实现
- **预训练模型**: 包含已训练的模型权重
- **数据集**: CUB-200-2011, COCONut, COCO Attributes
- **开发工具**: Jupyter Lab, TensorBoard, 调试工具

## 🚀 快速开始

### 1. 构建镜像

```bash
# 使用自动化脚本（推荐）
./build-docker.sh --export

# 或手动构建
docker build -t weak-supervised-cross-modal:latest .
```

### 2. 运行容器

```bash
# 交互式运行
docker run -it --gpus all \
  -p 8888:8888 -p 6006:6006 \
  -v $(pwd)/results:/workspace/results \
  weak-supervised-cross-modal:latest bash

# 直接训练
docker run -it --gpus all weak-supervised-cross-modal:latest train-cub --epochs 50

# 启动Jupyter
docker run -it --gpus all -p 8888:8888 weak-supervised-cross-modal:latest jupyter
```

### 3. 使用docker-compose（推荐）

```bash
# 启动主服务
docker-compose up weak-supervised-cross-modal

# 启动Jupyter Lab
docker-compose --profile jupyter up jupyter

# 后台训练
docker-compose --profile training up training
```

## 🔧 AutoDL部署

### 上传镜像

```bash
# 1. 本地导出镜像
docker save weak-supervised-cross-modal:latest | gzip > weak-supervised-cross-modal.tar.gz

# 2. 上传到AutoDL服务器

# 3. 在AutoDL上加载镜像
gunzip -c weak-supervised-cross-modal.tar.gz | docker load
```

### AutoDL运行

```bash
# 检查GPU
nvidia-smi

# 运行容器
docker run -it --gpus all \
  -p 8888:8888 -p 6006:6006 \
  -v /root/autodl-tmp:/workspace/results \
  weak-supervised-cross-modal:latest bash
```

## 📊 可用命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `train-cub` | 训练CUB数据集 | `train-cub --epochs 50` |
| `train-coco` | 训练COCO属性 | `train-coco --epochs 40` |
| `train-coconut` | 训练COCONut | `train-coconut --epochs 100` |
| `inference` | 运行推理 | `inference --model model.pth` |
| `jupyter` | 启动Jupyter Lab | `jupyter` |
| `tensorboard` | 启动TensorBoard | `tensorboard` |
| `bash` | 交互式shell | `bash` |

## 📁 目录结构

```
/workspace/
├── weak_supervised_cross_modal/    # 主要代码
│   ├── models/                     # 模型定义
│   ├── training/                   # 训练相关
│   ├── utils/                      # 工具函数
│   └── main.py                     # 主入口
├── data/                          # 数据集
│   ├── CUB_200_2011/              # CUB数据集
│   ├── coconut/                   # COCONut数据集
│   └── cocottributes-master/      # COCO属性数据集
├── checkpoints/                   # 模型检查点
├── results/                       # 训练结果
├── logs/                         # 日志文件
└── configs/                      # 配置文件
```

## 🎯 主要特性

- **GPU加速**: 支持NVIDIA GPU训练和推理
- **多数据集**: 支持CUB、COCO、COCONut数据集
- **可视化**: 集成Jupyter Lab和TensorBoard
- **持久化**: 结果和模型自动保存
- **易部署**: 一键部署到AutoDL服务器

## 📈 性能指标

根据之前的训练结果：

- **CUB数据集**: 训练精度 60-80%，验证精度 50-70%
- **COCO属性**: 40轮训练达到良好效果
- **COCONut**: 100轮训练，完整数据集支持

## 🛠️ 开发工具

### Jupyter Lab
- 访问地址: http://localhost:8888
- 预装所有依赖包
- 支持GPU加速计算

### TensorBoard
- 访问地址: http://localhost:6006
- 实时监控训练过程
- 可视化损失和指标

## 🔍 故障排除

### 常见问题

1. **GPU不可用**: 确保安装nvidia-container-toolkit
2. **内存不足**: 减少batch_size或增加--shm-size
3. **端口冲突**: 修改端口映射
4. **权限问题**: 使用sudo或调整文件权限

### 调试命令

```bash
# 查看容器日志
docker logs container_name

# 进入运行中的容器
docker exec -it container_name bash

# 监控资源使用
docker stats container_name
```

## 📞 技术支持

如需帮助，请检查：
1. `Docker部署指南.md` - 详细部署说明
2. `weak_supervised_cross_modal/快速启动指南.md` - 项目使用指南
3. 容器日志中的错误信息

## 🎉 优势

✅ **免环境配置**: 开箱即用的完整环境  
✅ **快速部署**: 一键部署到AutoDL服务器  
✅ **资源优化**: 多阶段构建，镜像大小优化  
✅ **易于使用**: 简化的命令行接口  
✅ **完整功能**: 训练、推理、可视化一体化  

---

**开始使用**: `./build-docker.sh --export` 然后上传到AutoDL服务器！
