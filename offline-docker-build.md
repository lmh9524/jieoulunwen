# 离线Docker构建方案

由于网络连接问题，无法直接从Docker Hub下载基础镜像。以下是几种解决方案：

## 🎯 推荐方案：使用便携式包

我已经为您创建了一个便携式项目包：`weak-supervised-cross-modal-package.zip` (1.2GB)

### 优势：
- ✅ 无需Docker，直接在AutoDL上运行
- ✅ 包含所有代码、模型和配置
- ✅ 自动化安装脚本
- ✅ 快速启动脚本
- ✅ 文件大小相对较小

### 使用方法：
1. 上传 `weak-supervised-cross-modal-package.zip` 到AutoDL
2. 解压：`unzip weak-supervised-cross-modal-package.zip`
3. 安装：`cd weak-supervised-cross-modal-package && ./install.sh`
4. 运行：`./quick_start.sh`

## 🔧 Docker方案（需要网络）

如果您有稳定的网络连接，可以尝试以下方法：

### 方案1：使用代理
```bash
# 配置Docker代理（如果有代理服务器）
docker build --build-arg HTTP_PROXY=http://proxy:port \
             --build-arg HTTPS_PROXY=http://proxy:port \
             -t weak-supervised-cross-modal:latest .
```

### 方案2：使用国内镜像源
修改Docker配置使用国内镜像源：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

### 方案3：分步构建
```bash
# 1. 先拉取基础镜像
docker pull python:3.9-slim

# 2. 然后构建项目镜像
docker build -t weak-supervised-cross-modal:latest .

# 3. 导出镜像
docker save weak-supervised-cross-modal:latest | gzip > weak-supervised-cross-modal.tar.gz
```

## 📦 包内容对比

| 方案 | 大小 | 优势 | 劣势 |
|------|------|------|------|
| 便携式包 | 1.2GB | 简单、快速、无需Docker | 需要手动安装依赖 |
| Docker镜像 | ~3-5GB | 完整环境、一致性好 | 需要网络、文件较大 |

## 🚀 AutoDL部署建议

### 推荐流程：
1. **上传便携式包**到AutoDL服务器
2. **解压并安装**：快速设置环境
3. **开始训练**：使用快速启动脚本

### 便携式包特性：
- 📁 完整项目代码
- 🤖 预训练模型 (best_*.pth)
- 📊 训练历史和结果
- 📖 完整文档
- 🛠️ 自动化脚本
- ⚡ 快速启动选项

## 💡 使用提示

### 在AutoDL上：
```bash
# 1. 上传并解压
unzip weak-supervised-cross-modal-package.zip
cd weak-supervised-cross-modal-package

# 2. 安装环境
chmod +x install.sh
./install.sh

# 3. 快速启动
chmod +x quick_start.sh
./quick_start.sh

# 4. 或手动运行
cd weak_supervised_cross_modal
python main.py --dataset cub --mode train --epochs 50
```

### 启动Jupyter：
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 启动TensorBoard：
```bash
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

## 🎯 总结

考虑到网络问题和部署效率，**强烈推荐使用便携式包方案**：

1. ✅ 文件已准备好：`weak-supervised-cross-modal-package.zip`
2. ✅ 大小合理：1.2GB（比Docker镜像小）
3. ✅ 部署简单：解压即用
4. ✅ 功能完整：包含所有必要组件
5. ✅ 自动化：一键安装和启动

这样可以避免Docker网络问题，同时保持部署的便利性！
