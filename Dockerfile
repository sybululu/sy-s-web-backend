# 使用 Python 3.11 镜像（torch/llama-cpp-python 兼容性最佳）
FROM python:3.11-slim

# 安装系统依赖（llama-cpp-python 编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 先安装 torch（CPU 版本，预编译 wheel，避免长时间编译）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 安装 llama-cpp-python（预编译 wheel，避免源码编译超时）
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 安装其余依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码到工作目录
COPY . .

# 暴露端口
EXPOSE 7860

# 启动 FastAPI 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
