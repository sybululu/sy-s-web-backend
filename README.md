# 隐私政策合规智能审查平台 - Python 后端

这是根据论文规范实现的 FastAPI 后端核心代码。

## 目录结构
- `app.py`: FastAPI 核心应用，包含了路由、Pydantic Schema、合规评分算法以及深度学习模型的骨架代码。
- `requirements.txt`: Python 依赖文件。

## 如何运行

1. 确保你安装了 Python 3.8+
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 启动服务：
   ```bash
   uvicorn app:app --reload
   ```
   或者直接运行：
   ```bash
   python app.py
   ```

4. 访问 API 文档：
   服务启动后，打开浏览器访问 `http://localhost:8000/docs` 即可看到自动生成的 Swagger UI，你可以在这里直接测试 `/api/v1/analyze` 和 `/api/v1/rectify` 接口。

## 核心逻辑说明
- **模型接入**：代码中已经预留了 `transformers` 和 `pymilvus` 的加载和调用骨架（被注释掉的部分）。当你训练好自己的模型后，只需取消注释并替换模型路径即可。
- **评分算法**：严格按照论文中的 `S = 100 - Σ (wi * vi) * 100` 实现，并内置了 12 项指标的权重。
- **当前状态**：目前 `mock_roberta_predict` 和 RAG 生成部分使用的是模拟数据，以便你现在就可以通过 `uvicorn` 跑通整个流程。
