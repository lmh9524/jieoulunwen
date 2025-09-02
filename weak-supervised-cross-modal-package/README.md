# 弱监督跨模态属性对齐项目 - 便携式包

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
