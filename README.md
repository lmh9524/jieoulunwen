# 弱监督解耦的跨模态属性对齐

## 项目概述

本项目实现了一个基于弱监督学习的跨模态属性对齐模型，主要用于图像属性识别和多任务分类。项目采用了多个创新技术模块，包括MAVD动态伪标签生成、CAL对比对齐策略、AFANet频域解耦、WINNER层级分解和CMDL轻量化正则化。

## 核心创新点

### 1. MAVD 动态伪标签生成
- 多专家动态路由机制
- 无监督解耦表示学习
- 自适应标签质量评估

### 2. CAL 对比对齐策略
- 跨模态特征对齐优化
- 对比学习损失函数
- 属性间关系建模

### 3. AFANet 频域解耦
- 高低频特征分离
- 频域注意力机制
- 多尺度特征融合

### 4. WINNER 层级分解
- 结构化语义生成
- 层级一致性约束
- 多层次属性表示

### 5. CMDL 轻量化正则化
- 基于互信息的属性解耦
- 轻量级正则化项
- MINE神经网络估计

## 支持的数据集

- **CelebA**: 人脸属性识别（40个属性，202,599张图像）
- **VAW**: 视觉属性词汇（620个属性）
- **COCO-Attributes**: 物体属性标注
- **CUB-200-2011**: 鸟类细粒度分类

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 至少8GB GPU内存

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/lmh9524/jieoulunwen.git
cd jieoulunwen

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

#### CelebA数据集（推荐）
确保CelebA数据集按以下结构组织：
```
/autodl-pub/data/CelebA/
├── img_align_celeba/          # 图像文件
├── annotations/
│   ├── list_attr_celeba.txt   # 属性标注
│   └── list_eval_partition.txt # 数据分割
```

### 3. 开始训练

#### Linux服务器（推荐）
```bash
# 设置执行权限
chmod +x train_celeba.sh

# 启动训练
./train_celeba.sh
```

#### 直接运行Python脚本
```bash
python train_celeba.py
```

## 项目结构

```
jieoulunwen/
├── weak_supervised_cross_modal/     # 核心模型代码
│   ├── config/                      # 配置文件
│   ├── models/                      # 模型架构
│   ├── training/                    # 训练相关
│   ├── data/                        # 数据加载器
│   └── utils/                       # 实用工具
├── train_celeba.py                  # CelebA训练脚本
├── train_celeba.sh                  # Linux执行脚本
├── requirements.txt                 # Python依赖
└── README.md                        # 项目文档
```

## 模型架构

### 主要组件

1. **特征提取器**: ResNet50预训练模型
2. **频域解耦模块**: AFANet高低频分离
3. **动态路由器**: MAVD多专家机制
4. **对比学习模块**: CAL跨模态对齐
5. **层级分解器**: WINNER结构化表示
6. **正则化器**: CMDL互信息约束

### 损失函数

- 分类损失（多任务）
- 对比学习损失
- 层级一致性损失
- 频域重构损失
- 互信息正则化损失

## 训练配置

### CelebA默认配置
- 批大小: 16
- 学习率: 1e-4
- 训练轮数: 50
- 图像尺寸: 224x224
- 混合精度训练: 启用

### 模型优化
- 自动混合精度 (AMP)
- 梯度裁剪
- 余弦退火学习率调度
- AdamW优化器

## 实验结果

### CelebA数据集
- 40个面部属性分类
- 多任务准确率: >90%
- 训练时间: ~10小时 (RTX 3070 Ti)

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 启用混合精度训练
   - 使用梯度累积

2. **数据集路径错误**
   - 检查配置文件中的data_path
   - 确保数据集目录结构正确

3. **依赖包版本冲突**
   - 使用虚拟环境
   - 严格按照requirements.txt安装

## 开发团队

- 研究方向：计算机视觉、多模态学习
- 联系方式：GitHub Issues

## 许可证

本项目仅用于学术研究目的。

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{weakly_supervised_cross_modal_2024,
  title={弱监督解耦的跨模态属性对齐},
  author={研究团队},
  year={2024},
  url={https://github.com/lmh9524/jieoulunwen}
}
```

## 更新日志

### v1.0.0 (2024-09-02)
- ✅ 完成CelebA数据集适配
- ✅ 实现全部5个创新模块
- ✅ 支持Linux服务器部署
- ✅ 混合精度训练优化
- ✅ 完整的训练和评估流程 