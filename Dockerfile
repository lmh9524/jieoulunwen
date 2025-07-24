# 弱监督跨模态属性对齐项目 Docker镜像
# 简化版本，避免网络问题

FROM python:3.9-slim

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements并安装依赖
COPY weak_supervised_cross_modal/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir jupyter jupyterlab tensorboard

# 创建必要的目录
RUN mkdir -p /workspace/weak_supervised_cross_modal \
    /workspace/data \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/results

# 复制项目代码
COPY weak_supervised_cross_modal/ /workspace/weak_supervised_cross_modal/

# 复制配置文件（如果存在）
COPY configs/ /workspace/configs/ 2>/dev/null || true

# 复制预训练模型（如果存在）
COPY *.pth /workspace/ 2>/dev/null || true
COPY checkpoints/ /workspace/checkpoints/ 2>/dev/null || true

# 复制数据集（重要文件）
COPY data/ /workspace/data/ 2>/dev/null || true

# 复制启动脚本
COPY docker-entrypoint.sh /workspace/docker-entrypoint.sh
RUN chmod +x /workspace/docker-entrypoint.sh

# 暴露端口
EXPOSE 8888 6006

# 设置入口点
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
CMD ["bash"]
