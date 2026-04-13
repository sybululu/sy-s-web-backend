---
title: 隐私政策合规审查
emoji: 🔒
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "0.0.5"
python_version: "3.9"
app_file: app.py
pinned: false
---

# 隐私政策合规审查系统

基于 BERT-MoE 和 RAG 的隐私政策合规审查系统，支持 12 类违规检测。

## 部署说明

此仓库托管于 HuggingFace Spaces：https://huggingface.co/spaces/sybululu/privacy-policy-checker

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
