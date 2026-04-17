# 使用 Python 3.11 镜像（torch/llama-cpp-python 兼容性最佳）
FROM python:3.11-slim

# 安装系统依赖（llama-cpp-python 源码编译需要 cmake + build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 1. 安装 torch（CPU 预编译 wheel，~18秒完成）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 llama-cpp-python（纯 CPU 编译优化）
#    abetlen 索引对 0.3.x 无 manylinux_2_28 预编译包，必须源码编译
#    CMAKE_ARGS 禁用所有非 CPU 后端，大幅缩减编译时间
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -LLAMA_CUDA=OFF -LLAMA_METAL=OFF -LLAMA_VULKAN=OFF -LLAMA_ACCELERATE=OFF -LLAMA_CLBLAST=OFF" \
    pip install --no-cache-dir llama-cpp-python==0.2.90 \
    --no-binary :all: \
    --config-settings="cmake.args=-DLLAMA_NATIVE=OFF"

# 3. 安装其余 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码到工作目录
COPY . .

# 暴露端口
EXPOSE 7860

# 启动 FastAPI 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
