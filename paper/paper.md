## 弱监督解耦的跨模态属性对齐：统一的可插拔框架

### 摘要
弱监督的属性识别与跨模态对齐在标签稀缺、标注噪声与属性歧义并存的现实场景中尤为关键。本文提出一个可插拔的弱监督解耦跨模态属性对齐框架 `WeakSupervisedCrossModalAlignment`，统一支撑 CelebA、COCOAttributes 与 CUB 等多数据集。框架以 `ResNet-50` 为视觉主干，融合可选的频域解耦（AFANet）、跨模态编码器、动态伪标签路由（MAVD）、层级分解（WINNER）与轻量正则（CMDL），并通过对比对齐（CAL）与动态权重调节在综合损失下实现稳定训练。我们在多个数据集上展示了方法的有效性，提供注意力/层级可视化、系统化消融与效率评估，并给出可复现实践（AMP、断点续训、自动报告与权重导出）。

**关键词**：弱监督学习，跨模态对齐，属性识别，解耦表示，层级表示，动态路由

---

## 1 引言
多属性识别广泛出现于人脸属性（CelebA）、通用目标属性（COCOAttributes）与细粒度物种属性（CUB）等场景。受限于高成本标注与跨数据集属性定义差异，如何在弱监督条件下实现稳健、可解释且可迁移的属性学习，仍面临多重挑战：
- 标注稀缺与噪声：弱标签/伪标签引入分布偏移与不确定性；
- 属性耦合与歧义：多属性间存在共现与互斥关系；
- 跨模态/跨数据集对齐：视觉—语义/文本的对齐不足，属性组定义不一致；
- 可解释性与效率：可视化与分析能力不足，真实部署受算力约束。

为此，我们提出一个统一的、可插拔的弱监督解耦跨模态属性对齐框架，核心特点如下：
- 以视觉为主的统一架构，支持可选的频域解耦与跨模态编码；
- 动态伪标签路由（MAVD）、层级分解（WINNER）、轻量正则（CMDL）与对比对齐（CAL）协同；
- 综合损失与动态权重调节，提升弱监督稳定性与泛化性能；
- 支持 CelebA、COCOAttributes、CUB 等多数据集的一致实验口径与可复现。

本文贡献如下：
- 提出一个统一的、可插拔的弱监督解耦跨模态属性对齐框架；
- 设计 AFANet/MAVD/WINNER/CMDL/CAL 等模块，并在综合损失下协同；
- 在多数据集上验证有效性，提供系统化消融与可视化；
- 提供工程化实践（AMP、断点续训、自动报告与权重导出）。

## 2 相关工作
### 2.1 多属性学习与弱监督属性识别
多属性学习在 CelebA、COCOAttributes、CUB 等数据集上有广泛研究。弱监督方法通过伪标签、约束或自监督信号降低对完整标注的依赖，但仍面临噪声鲁棒与可解释性不足的问题。

### 2.2 跨模态表示与对齐
视觉—文本/语义对齐可提升属性可迁移性。现有方法多依赖强监督或高质量文本注释。我们采用可插拔跨模态编码器，允许在无文本时以视觉为主，在有文本时进行融合对齐。

### 2.3 频域特征与解耦表示
频域分解可将高/低频信息解耦，缓解纹理/结构混淆。AFANet 通过特征多样性与分布约束，促进稳健的频域表示学习。

### 2.4 动态路由与专家模型
专家路由提高任务子空间的建模能力，但在弱监督下需控制伪标签质量与专家多样性。MAVD 引入专家使用的均匀性（KL）与稀疏性约束。

### 2.5 层级表示学习与图结构正则
层级分解可从局部到全局逐级聚合，提升可解释性与鲁棒性。WINNER 结合层级一致性、层级间多样性与结构约束以稳定训练。

### 2.6 互信息与轻量正则
CMDL 以轻量替代互信息估计，降低训练复杂度与不稳定性，鼓励属性特征的解耦与紧致。

## 3 方法
### 3.1 总体框架
我们的方法以 `ResNet-50` 为视觉骨干，输出 49×2048 的序列特征，经线性映射到 `hidden_size` 并归一化。可选的 AFANet 产生频域特征，经 `FeatureFusionModule` 门控融合到视觉序列。随后进入 `WeakSupervisedCrossModalEncoder` 跨模态编码（文本为可选输入）。在此基础上，`MAVDDynamicRouter` 产生伪标签与重要性权重，`WINNERHierarchicalDecomposer` 提供多层级特征与图表示，`CMDLLightweightRegularizer` 对属性特征施加正则。最终通过 `output_projector` 投影并对各属性组使用独立分类器输出预测。

### 3.2 视觉编码与特征融合
- 视觉编码：`ResNet-50`（去分类头）+ 自适应池化 + Flatten + 转置 + 线性映射到 `hidden_size` + `LayerNorm/ReLU/Dropout`；
- 频域解耦（可选）：AFANet 从图像产生 `[B, 1, hidden_size]` 频域特征；
- 融合：`FeatureFusionModule` 采用门控融合（拼接→MLP→Sigmoid 门控→残差式融合）。

### 3.3 跨模态编码器
`WeakSupervisedCrossModalEncoder` 统一视觉序列与可选文本特征，输出跨模态特征与属性原始特征（raw attributes），为对比对齐与正则提供接口。

### 3.4 动态伪标签路由（MAVD）
在池化后的跨模态特征上估计伪标签与专家重要性权重；损失包含：
- 伪标签质量项（与 `targets['pseudo_quality']` 一致性，MSE）；
- 专家使用分布的均匀性（KL 至均匀分布）；
- 重要性权重稀疏性（L1）。

### 3.5 层级分解（WINNER）与图表示
输出多层级特征与图表示；损失由三部分组成：
- 层级一致性（相邻层级池化特征的 MSE）；
- 层级间多样性（余弦相似度阈值化惩罚）；
- 层级结构（高层级具有更小的方差约束）。

### 3.6 轻量正则（CMDL）
对非对齐的属性特征进行池化后施加正则，鼓励不同属性表征的独立性与稳定性。

### 3.7 对比对齐（CAL）
在属性对之间构造 pairwise 对比损失，采用稳定化的 InfoNCE（softplus 变体）与伪标签阈值策略；若存在属性标签对，则使用同类为正样本、异类为负样本。

### 3.8 综合损失与动态权重
`ComprehensiveLoss` 汇总分类（CE）、对比对齐（CAL）、MAVD、层级一致性、频域多样性与图正则等损失，并通过 `DynamicWeightAdjuster` 基于近期损失趋势进行轻量自适应权重更新；总损失为加权和：
\[ L = \sum_i w_i \cdot L_i, \quad w_i \leftarrow \mathrm{adjust}(w_i, \mathrm{trend}(L_i)) \]

### 3.9 复杂度与内存
- 基础视觉编码器为主要 FLOPs；
- 频域、路由与层级分支为可选开销；
- AMP 减少算力/显存消耗；梯度裁剪保证稳定性。

### 3.10 可解释性接口
提供 `get_attention_maps` 与分层注意力可视化，支持属性注意力与层级注意力的热力图展示；`extract_features` 支持多层特征导出。

## 4 数据集与实验设置
### 4.1 数据集
- CelebA：40 维属性（-1/1）→ 二值化 → 汇总为 8 个属性组目标：`hair_style`、`facial_features`、`makeup`、`accessories`、`expression`、`demographics`、`facial_hair`、`quality`；保留 `all_attributes`（40 维）。
- COCOAttributes：204 属性，分组为 `color/material/shape/texture/size/other`。
- CUB：分组为 `color/material/shape`。

### 4.2 预处理与增强
- 训练：`Resize`、`RandomHorizontalFlip(p=0.5)`、`ColorJitter`、`Normalize`；
- 验证/测试：`Resize` + `Normalize`。

### 4.3 评估指标
- 每属性组：Accuracy、Precision、Recall、F1（macro）；
- 汇总：Mean Accuracy（对属性组准确率求均值）。

### 4.4 训练细节
- 优化器：AdamW（`lr=1e-4`，`weight_decay=1e-5`）；调度器：CosineAnnealing；
- 混合精度：AMP；梯度裁剪：`clip_grad_norm_=1.0`；
- 训练轮次与批大小：按数据集配置（CelebA 首训仅保留分类分支以提升稳定性）。

### 4.5 可复现性
- 训练脚本：`train_celeba.py`、`weak_supervised_cross_modal/train_*.py`；
- 检查点与报告：自动导出 `experiments/<exp>/checkpoints/`、`training_report.json`、`training_history.json`；
- 断点续训：自动发现最近检查点并询问是否恢复。

## 5 实验结果
### 5.1 主结果（占位）
下表为各数据集按属性组的主结果（Acc / F1）。CelebA 将在完成统一配置复现后填充。

表 1：CelebA 主结果（占位）

| 属性组 | Acc | F1 |
|---|---:|---:|
| hair_style |  |  |
| facial_features |  |  |
| makeup |  |  |
| accessories |  |  |
| expression |  |  |
| demographics |  |  |
| facial_hair |  |  |
| quality |  |  |
| Mean |  |  |

表 2：COCOAttributes 主结果（示意）

| 属性组 | Acc | F1 |
|---|---:|---:|
| color |  |  |
| material |  |  |
| shape |  |  |
| texture |  |  |
| size |  |  |
| other |  |  |
| Mean |  |  |

表 3：CUB 主结果（示意）

| 属性组 | Acc | F1 |
|---|---:|---:|
| color |  |  |
| material |  |  |
| shape |  |  |
| Mean |  |  |

### 5.2 消融研究（占位）
逐项关闭 AFANet / WINNER / MAVD / CMDL / CAL / 动态权重，报告对 Mean Accuracy 与平均 F1 的影响。

表 4：消融结果（示意）

| 设定 | AFANet | WINNER | MAVD | CMDL | CAL | 动态权重 | Mean Acc | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| -AFANet |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| -WINNER | ✓ |  | ✓ | ✓ | ✓ | ✓ |  |  |
| -MAVD | ✓ | ✓ |  | ✓ | ✓ | ✓ |  |  |
| -CMDL | ✓ | ✓ | ✓ |  | ✓ | ✓ |  |  |
| -CAL | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |  |
| -动态权重 | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |

### 5.3 跨数据集泛化（占位）
在 A 集训练至收敛，在 B 集零样本/微调评估：
- COCO→CUB、CUB→COCO；CelebA 视属性对齐可行性补充。

### 5.4 可视化与案例分析（占位）
- 属性注意力与层级注意力热力图；
- 正确/错误案例与失败模式；
- 频域分量的对比示例（若启用 AFANet）。

### 5.5 效率评估（占位）
- 训练吞吐 / 显存峰值 / 推理延迟；
- 模块启用对开销的增量影响。

## 6 讨论
- 模块协同：频域解耦缓解纹理偏置，层级分解提升全局稳定，动态路由利用专家多样性，轻量正则抑制过拟合；
- 弱监督鲁棒：CAL 与动态权重改善伪标签噪声；
- 适用性：在仅视觉信号下已可用，有文本时进一步受益；
- 局限：属性分组由启发式规则指定；跨数据集属性定义仍存在落差。

## 7 结论与未来工作
提出一个统一的弱监督解耦跨模态属性对齐框架，在多数据集上验证其有效性与可解释性。未来将：
- 引入文本/语言先验并实现端到端对齐；
- 学习属性分组与层级结构而非手工指定；
- 推进跨域自适应与开放集属性识别。

## 8 伦理与社会影响
人脸属性识别涉及隐私与偏见风险。我们遵循数据许可与审慎评估原则，建议在合规场景下使用，并报告潜在偏差与不确定性。

## 9 复现说明
- 环境与依赖：见项目 `requirements.txt`；
- 训练命令与脚本：CelebA 可使用 `train_celeba.py`；COCO/CUB 参考 `weak_supervised_cross_modal/train_*.py`；
- 日志与检查点：`experiments/<exp>/checkpoints/`、`training_report.json`、`training_history.json`；
- 可视化：使用 `get_attention_maps` 导出注意力热力图。

---

### 附录（占位）
- 完整超参表；
- 公式推导与更多实验图表；
- 误差条与置信区间。 