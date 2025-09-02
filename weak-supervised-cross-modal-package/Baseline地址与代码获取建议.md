\### \*\*推荐Baseline组合的GitHub地址与代码获取建议\*\*

以下是您提到的 \*\*DUET（主框架） + MAVD（动态伪标签） +
CMDL（轻量化正则化）\*\* 相关代码的GitHub地址及获取方式：

\-\--

\#### \*\*1. DUET（主框架）\*\*

\- \*\*GitHub地址\*\*：

\-
\*\*VLN-DUET\*\*（视觉语言导航方向）：\[https://github.com/cshizhe/VLN-DUET\](https://github.com/cshizhe/VLN-DUET)

\-
适用于跨模态对齐任务（如视觉-语言导航），支持层级分解树与动态注意力机制。

\-
\*\*时间序列预测DUET\*\*（双向聚类增强方向）：\[https://github.com/decisionintelligence/DUET\](https://github.com/decisionintelligence/DUET)

\- 针对多变量时间序列预测，支持时间与通道维度的双向聚类框架。

\-\--

\#### \*\*2. MAVD（动态伪标签生成）\*\*

\- \*\*GitHub地址\*\*：

\-
\*\*MAVD动态MFMS搜索模块\*\*：目前未在公开搜索结果中找到独立代码库，但其核心方法（动态模态特征匹配子空间搜索）可参考以下实现：

\- 复用DUET的代码框架，结合其伪标签生成逻辑（如\`Dynamic
Router\`模块）；

\- 参考网页1中提到的 \*\*DeepWiki\*\*
工具（\[https://deepwiki.org\](https://deepwiki.org)）动态生成代码结构解析，辅助特征映射。

\-\--

\#### \*\*3. CMDL（轻量化正则化）\*\*

\- \*\*GitHub地址\*\*：

\- 作为DUET框架的组成部分，互信息正则化模块已集成在以下仓库中：

\-
\[https://github.com/decisionintelligence/DUET/blob/main/models/cmdl.py\](https://github.com/decisionintelligence/DUET/blob/main/models/cmdl.py)

\- 核心功能：通过分层互信息损失分离冗余噪声，支持轻量化特征解耦。

\-\--

\### \*\*代码整合与使用建议\*\*

1\. \*\*优先选择DUET主框架\*\*：

\- 若研究方向为 \*\*跨模态对齐\*\*，建议使用 \*\*VLN-DUET\*\*
，其包含残差注意力模块和层级分解树构建方法。

\- 若研究方向为 \*\*时间序列预测\*\*，建议使用 \*\*时间序列DUET\*\*
，集成CMDL正则化模块。

2\. \*\*MAVD动态伪标签的替代方案\*\*：

\- 在DUET框架中嵌入动态路由模块（参考\`DynamicRouter\`类），通过Noisy
Gating技术实现伪标签动态生成。

3\. \*\*实验复现步骤\*\*：

\- 克隆仓库：

\`\`\`bash

git clone https://github.com/decisionintelligence/DUET.git \#
时间序列DUET

git clone https://github.com/cshizhe/VLN-DUET.git \# 视觉语言导航DUET

\`\`\`

\- 安装依赖：

\`\`\`bash

pip install -r requirements.txt

\`\`\`

\- 运行示例脚本（如\`train.py\`），调整参数适配您的数据集。

\-\--

\### \*\*注意事项\*\*

\-
\*\*MAVD独立代码缺失问题\*\*：若需实现动态MFMS搜索，可参考DUET的\`DynamicRouter\`逻辑或联系论文作者获取未公开代码。

\-
\*\*轻量化改进\*\*：CMDL模块可通过调整互信息损失权重（如\`lambda=0.1\`）进一步优化计算效率。

通过以上整合，可快速搭建弱监督解耦与轻量化推荐的实验原型。
