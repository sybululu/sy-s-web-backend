# 使用官方 Python 3.9 轻量级镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码到工作目录
COPY . .

# 暴露 Hugging Face Spaces 默认端口 7860
EXPOSE 7860

# 启动 FastAPI 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
