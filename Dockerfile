# 隐私政策合规审查系统 - 后端
#
# 默认使用 HuggingFace Inference API（极速部署，~1分钟构建）
# 如需本地 GGUF 模式，设置环境变量 LLM_MODE=local 并使用 Dockerfile.local
FROM python:3.11-slim

# 安装最小系统依赖（仅 torch 编译需要，不再需要 cmake）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. 安装 torch（CPU 预编译 wheel，~18秒）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装其余依赖（不含 llama-cpp-python，使用 HF Inference API）
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# 环境变量说明:
#   LLM_MODE=api        (默认) 使用 HuggingFace Inference API，需设置 HF_TOKEN
#   LLM_MODEL_ID         默认 microsoft/Phi-4-mini-instruct，可替换为其他模型
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
