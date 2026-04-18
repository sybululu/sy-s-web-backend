"""
隐私政策合规审查 API
整合了 RAG 架构的法律知识库检索

整改生成模型支持三种模式（通过环境变量 LLM_MODE 切换）:
  - "github"  : GitHub Models (默认，推荐，免费 Phi-4 Mini，OpenAI 兼容接口)
  - "local"   : 本地 GGUF (llama-cpp-python + Phi-4 Mini Q6_K)
  - "hf"      : HuggingFace Inference API (需 Token 有 Inference 权限)

Token 分工：
  - HF_TOKEN    : 仓库通行证，用于下载模型权重（RoBERTa、嵌入模型、私有 .ckpt）
  - GITHUB_TOKEN: 大脑通行证，用于调用 GitHub Models 上的 Phi-4 Mini 推理
"""
from __future__ import annotations

import json
import os
import re
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

import torch
from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download, InferenceClient

from models import User, Project, get_db, init_db, Article, RetrievedChunk
from auth import router as auth_router, get_current_user

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# LLM 模式配置
# ==========================================
LLM_MODE = os.getenv("LLM_MODE", "github")   # "github" | "local" | "hf"
HF_TOKEN = os.getenv("HF_TOKEN", "")          # HF 仓库通行证：下载模型权重（必留！）
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # GitHub Models 通行证：Phi-4 推理
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Phi-4-mini-instruct")  # GitHub Models 上的模型 ID

# ==========================================
# 导入 RAG 模块
# ==========================================
import sys as _sys
from pathlib import Path as _Path

# 确保 src/ 能被正确识别为包（兼容 HF Space 脚本模式）
_project_root = str(_Path(__file__).resolve().parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

RAG_AVAILABLE = False
legal_kb_loader: Optional[Any] = None
vector_store: Optional[Any] = None
retriever: Optional[Any] = None

try:
    from src.loader import LegalKBLoader
    from src.store import VectorStore
    from src.search import Retriever
    from src.config import get_config
    RAG_AVAILABLE = True
    logger.info("RAG 模块加载成功")
except ImportError as e:
    logger.warning(f"RAG 模块加载失败: {e}，将使用静态配置降级")

# ==========================================
# 模型加载 (HuggingFace Hub)
# ==========================================
print("正在从 HuggingFace Hub 加载模型，首次运行需要下载...")

import torch.nn as nn
# BertModel 已在文件顶部 from transformers import AutoTokenizer, BertModel 导入

# 1. 加载 RoBERTa 风险分类模型（自定义模型类，完全复现训练时结构）
# 训练代码中分类层命名为 self.fc（非标准库的 self.classifier），
# 因此必须自定义模型类来匹配 checkpoint 的键名空间。
class CustomBertMoeModel(nn.Module):
    """完全复现训练代码的模型结构：BertModel + fc(768→11)"""

    def __init__(self):
        super(CustomBertMoeModel, self).__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # 与训练代码一致：num_classes=11，分类层命名为 fc
        self.fc = nn.Linear(768, 11)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        前向传播：与训练代码结构完全一致。

        注意：必须接受 **kwargs，因为 transformers 的 tokenizer 在调用 model(**inputs)
        时会传入 token_type_ids 等额外关键字参数。训练时的 forward 只接收位置参数
        并通过索引取值（x[0], x[2]），生产环境需要兼容这种多参数传入方式。
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch, 768]
        return self.fc(pooled_output)          # [batch, 11]


def load_trained_model(ckpt_path: str) -> CustomBertMoeModel:
    """加载微调 checkpoint 到自定义模型结构"""
    model = CustomBertMoeModel()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 仅需移除 "model." 前缀（训练时用 DataParallel 或类似包装），
    # 无需做 "fc" → "classifier" 的重映射，因为类结构已完全匹配
    cleaned_state_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items()
    }

    # strict=True：由于类结构与训练完全一致，所有键应精确匹配
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    if missing:
        logger.warning(f"RoBERTa 缺失键: {missing}")
    if unexpected:
        logger.warning(f"RoBERTa 多余键: {unexpected}")

    model.eval()
    return model


# 初始化 tokenizer 和自定义模型
tokenizer_roberta = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

roberta_ckpt_path = hf_hub_download(
    repo_id="sybululu/bert-moe",
    filename="multi_classification_bertmoe.ckpt"
)
model_roberta = load_trained_model(roberta_ckpt_path)
print("✅ RoBERTa 分类模型加载完成（CustomBertMoeModel + fc 分类头）")

# 2. 加载整改生成模型（支持三种模式）
llm = None           # type: ignore  # HF InferenceClient (hf 模式)
llm_github = None    # type: ignore  # OpenAI 兼容客户端 (github 模式)

if LLM_MODE == "local":
    # 本地模式：使用 llama-cpp-python 加载 GGUF
    try:
        from llama_cpp import Llama as LocalLlama
        phi4_gguf_path = hf_hub_download(
            repo_id="tensorblock/Phi-4-mini-instruct-GGUF",
            filename="Phi-4-mini-instruct-Q6_K.gguf"
        )
        llm = LocalLlama(
            model_path=phi4_gguf_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        print(f"✅ Phi-4 Mini 本地模式加载完成 (GGUF, Q6_K)")
    except Exception as e:
        logger.warning(f"本地 GGUF 模式加载失败，回退到 github 模式: {e}")
        LLM_MODE = "github"

if LLM_MODE == "github":
    # GitHub Models 模式：OpenAI 兼容接口，免费调用 Phi-4 Mini
    if not GITHUB_TOKEN or GITHUB_TOKEN == "your-github-token-here":
        logger.warning("GITHUB_TOKEN 未配置或仍为默认值，回退到 HF Inference API 模式")
        LLM_MODE = "hf"
    else:
        try:
            from openai import OpenAI
            llm_github = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=GITHUB_TOKEN,
            )
            print(f"✅ 整改生成模型: GitHub Models API 模式 (model={LLM_MODEL_ID})")
        except ImportError:
            logger.warning("openai 包未安装，回退到 HF Inference API 模式")
            LLM_MODE = "hf"
        except Exception as e:
            logger.warning(f"GitHub Models 初始化失败，回退到 HF Inference API 模式: {e}")
            LLM_MODE = "hf"

if LLM_MODE == "hf":
    # HF Inference API 模式（需 HF_TOKEN 有 Inference 权限，否则 403）
    llm = InferenceClient(model=LLM_MODEL_ID, token=HF_TOKEN or None)
    print(f"⚠️ 整改生成模型: HF Inference API 模式 (model={LLM_MODEL_ID}) — 需确保 Token 有 Inference 权限")

# ==========================================
# RAG 组件初始化
# ==========================================
def initialize_rag():
    """初始化 RAG 组件"""
    global legal_kb_loader, vector_store, retriever
    
    if not RAG_AVAILABLE:
        logger.warning("RAG 模块不可用，跳过初始化")
        return
    
    try:
        config = get_config()
        legal_kb_loader = LegalKBLoader(config.knowledge_dir)
        vector_store = VectorStore(
            embedding_model=config.embedding_model,
            persist_path=config.vector_store_path
        )
        retriever = Retriever(loader=legal_kb_loader, vector_store=vector_store)
        retriever.initialize()
        logger.info("RAG 组件初始化完成")
    except FileNotFoundError:
        # 知识库目录不存在（HF Space 无预置知识库），正常降级
        logger.info("知识库目录不存在，RAG 功能降级为静态配置（部署后可后续上传知识库启用）")
    except Exception as e:
        # FAISS/嵌入模型加载等意外错误
        logger.error(f"RAG 初始化异常（非预期）: {type(e).__name__}: {e}")
        # 不阻断主程序继续运行

# ==========================================
# FastAPI 应用设置
# ==========================================
app = FastAPI(title="隐私政策合规审查 API")

# CORS 配置（前端 Vite dev server 默认端口 5000，Cloudflare Pages 生产域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sy-s-web.pages.dev",   # Cloudflare Pages 生产环境
        "http://localhost:5000",        # Vite 开发环境
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
init_db()

# 注册认证路由
app.include_router(auth_router)

# ==========================================
# 合规指标体系与权重定义（唯一权威数据源：violation_config.py）
# ==========================================
from violation_config import INDICATORS, ID_TO_INDICATOR, ID_TO_HINT
from src.mapper import map_to_12_classes

# 创建快捷访问列表（保持向后兼容）
INDICATOR_KEYS = list(INDICATORS.keys())

# RoBERTa 11 类原始类别名称（与模型训练时输出维度一一对应）
CLASS_NAMES = [
    "数据收集", "权限获取", "共享转让", "使用目的", "存储方式",
    "安全销毁", "特殊人群", "权限管理", "联系方式", "政策变更", "停止运营",
]

# ==========================================
# Pydantic 数据模型定义 (Schema)
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    source_type: Optional[str] = "text"
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('文本不能为空')
        return v

class AnalyzeResponse(BaseModel):
    id: str
    name: str
    score: float
    risk_level: str
    violations: List[dict]
    sentence_results: List[dict] = Field(default_factory=list, description="每条句子的原始分类结果（测试集式明细）")

class RectifyRequest(BaseModel):
    original_snippet: str
    violation_type: str
    mode: str = Field(default="rewrite", description="整改模式: summary(摘要建议) | rewrite(完整改写)")
    legal_basis: Optional[str] = Field(None, description="前端传入的法律依据（可选，后端优先使用RAG检索结果）")

class UrlRequest(BaseModel):
    url: str

# ==========================================
# 风险等级阈值（与前端 violation-config.ts 对齐）
# ==========================================
# 审查级：基于总分
SCORE_THRESHOLD_LOW_RISK = 70     # score >= 70 → 低风险
SCORE_THRESHOLD_HIGH_RISK = 40    # score < 40 → 高风险（40~70 中等风险）

# ==========================================
# 辅助函数
# ==========================================
# 标题行正则模式（匹配后跳过 RoBERTa 分类，避免标题被误判为违规）
_HEADING_PATTERNS = re.compile(
    r'^('
    # 中文序号标题: 一、/ 二、/ （一）/ 1\. / 1、/ 第[一二三四五六七八九十\d]+[章节条款部分]
    r'[一二三四五六七八九十\d]+[\.\、\s]*(?:节|章|条款|部分|)'
    r'|[一二三四五六七八九十]+[\.\、]'
    # 英文/数字序号: 1. / 1) / (1)
    r'|^\d+[\.\)\s]+'
    # 常见隐私政策标题关键词（无实际行为的纯主题短语）
    r'|^(?:信息收集|信息使用|信息共享|数据安全|用户权利|Cookie|未成年人|'
    r'联系我们|政策更新|生效时间|适用范围|定义|总则|附则|修订记录'
    r')'
    # 纯标题短语（无谓语动词的名词串，≤20字且不含"我们/将/会/可以"等行为词）
    r'|^.{2,20}$'
    r')',
    re.UNICODE
)

# 行为动词列表：句子中若包含这些词，说明是有实际内容的正文而非标题
_ACTION_KEYWORDS = {'收集', '使用', '共享', '转让', '存储', '销毁', '删除', '访问',
                   '查阅', '更正', '撤回', '同意', '授权', '告知', '提供', '保留',
                   '处理', '获得', '采取', '发送', '接收', '加密', '匿名化',
                   '分享', '公开', '披露', '出售', '允许', '承诺', '保证'}


def is_likely_heading(sentence: str) -> bool:
    """
    判断一个句子是否为标题/小标题（非实质性条款内容）

    标题特征：
    1. 匹配常见标题格式（序号、章节名、纯主题短语）
    2. 较短（≤30字）且不包含行为动词
    3. 不含"我们/我方/本公司"等主体声明

    Returns:
        True → 是标题，应跳过分类
        False → 是正文，需要分类
    """
    s = sentence.strip()

    # 长度检查：过长的句子不太可能是标题
    if len(s) > 35:
        return False

    # 强标题模式：序号开头 / 章节名 / 常见标题关键词 → 直接判定为标题
    # 这些模式优先级最高，即使包含动作词也视为标题（如"一、信息收集"）
    _STRONG_HEADING = re.compile(
        r'^('
        r'[一二三四五六七八九十]+[\.\、]'           # 一、/ 二、/
        r'|[一二三四五六七八九十\d]+[\.\、\s]*(?:节|章|条款|部分)'  # 1.1 第一章
        r'|^\d+[\.\)\s]+'                           # 1. / 1) /
        r'|^(?:信息收集|信息使用|信息共享|数据安全|用户权利|Cookie|未成年人|'
        r'联系我们|政策更新|生效时间|适用范围|定义|总则|附则|修订记录)'
        r')',
        re.UNICODE
    )
    if _STRONG_HEADING.match(s):
        return True

    # 包含行为动词 → 很可能是正文（标题一般只列主题不写动作）
    has_action = any(kw in s for kw in _ACTION_KEYWORDS)

    # 包含主体声明词 → 正文特征
    has_subject = any(w in s for w in ('我们', '我方', '本公司', '本应用', '本平台',
                                         '将', '会', '可以', '可能', '应当', '需要',
                                         '用户', '您', '您的'))

    if has_subject and has_action:
        return False  # 有主体+有动作 = 正文

    # 仅含行为动词（无明确主体）→ 也视为正文（如"用户有权查阅..."）
    if has_action:
        return False

    # 弱标题模式：纯短文本（无动词无主体）
    if _HEADING_PATTERNS.match(s):
        return True

    return False


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'[。；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def roberta_predict(sentence: str) -> Dict[str, Any]:
    """
    RoBERTa 预测：11 类模型输出 → 通过 mapper 映射到 12 类违规类型

    返回完整分类明细（供 sentence_results 展示）:
    {
        "mapped": {violation_id: probability},   # 映射后的违规ID及概率，如 {"I1": 0.82}
        "raw_probs": [0.12, 0.85, ...],           # 11类原始sigmoid概率
        "max_class_idx": 2,                        # 最高概率类别索引(0~10)
        "max_prob": 0.85,                         # 最高类别概率值
        "confidence": 3.42,                       # 置信度(logits最高-次高差值)
        "class_name": "共享转让",                   # 最高类别中文名
    }
    无违规时 mapped 为空 {}
    """
    inputs = tokenizer_roberta(sentence, return_tensors="pt", truncation=True, max_length=150)
    with torch.no_grad():
        outputs = model_roberta(**inputs)
        # CustomBertMoeModel 直接返回 logits tensor（非 SequenceClassifierOutput 对象）
        if isinstance(outputs, torch.Tensor):
            logits = outputs.squeeze()
        else:
            # 兼容标准库输出格式（以防未来切回 AutoModelForSequenceClassification）
            logits = outputs.logits.squeeze()
        # 11 类 sigmoid 概率
        probs = torch.sigmoid(logits).tolist()

    if not isinstance(probs, list):
        probs = [probs]

    # 计算置信度：logits 最高值与次高值的差值（差越大越确信）
    if isinstance(logits, torch.Tensor) and logits.dim() == 1:
        sorted_logits, _ = torch.sort(logits, descending=True)
        confidence = (sorted_logits[0] - sorted_logits[1]).item() if len(sorted_logits) > 1 else sorted_logits[0].item()
    else:
        confidence = None

    # 11 类 → 12 类违规 ID 多标签映射（置信度 + 概率双重约束）
    detected_ids = map_to_12_classes(probs, confidence=confidence)

    # 原始 11 类最高概率信息
    max_idx = probs.index(max(probs)) if probs else -1
    max_prob = max(probs) if probs else 0.0

    # 取最高概率作为映射结果的置信度
    mapped_result = {vid: max_prob for vid in detected_ids} if detected_ids else {}

    # 日志：逐句分类明细（方便后端调试查看）
    logger.info(
        f"句子: {sentence[:40]}{'...' if len(sentence)>40 else ''} "
        f"| 类别: {CLASS_NAMES[max_idx] if 0 <= max_idx < len(CLASS_NAMES) else max_idx} "
        f"| 概率: {max_prob:.4f} | 置信度: {f'{confidence:.2f}' if confidence is not None else 'N/A'} "
        f"| 违规: {list(mapped_result.keys()) or '无'}"
    )

    return {
        "mapped": mapped_result,
        "raw_probs": [round(p, 6) for p in probs],
        "max_class_idx": max_idx,
        "max_prob": round(max_prob, 6),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "class_name": CLASS_NAMES[max_idx] if 0 <= max_idx < len(CLASS_NAMES) else f"未知({max_idx})",
    }

def get_legal_basis_from_rag(violation_type: str, context: Optional[str] = None) -> Dict[str, any]:
    """
    使用 RAG 检索获取法律依据（返回结构化数据：引用列表 + 正文列表）

    Args:
        violation_type: 违规类型ID，如 "I1"
        context: 违规上下文描述

    Returns:
        {
            "reference": "《个人信息保护法》第六条；网络安全法第四十一条",  # 逗号分隔用于展示
            "references": [                                              # 结构化列表
                {"law": "个人信息保护法", "article": "第六条", "content": "..."},
                {"law": "网络安全法", "article": "第四十一条", "content": "..."},
            ],
            "content": "处理敏感个人信息应当取得个人的单独同意..."          # 用于 prompt 注入（拼接所有正文）
        }
    """
    default_ref = INDICATORS.get(ID_TO_INDICATOR.get(violation_type, ""), {}).get("legal_basis", "《个人信息保护法》")

    if not RAG_AVAILABLE or retriever is None:
        return {"reference": default_ref, "references": [], "content": ""}

    try:
        results = retriever.retrieve_by_violation_type(violation_type, context=context, top_k=5)
        if results:
            references = []
            prompt_contents = []   # 注入 prompt 用，每条截断 200 字
            for r in results:
                ref = r.law_reference if hasattr(r, 'law_reference') else f"《{r.law}》{r.article_number}"
                full_content = r.content or ""
                references.append({
                    "law": getattr(r, 'law', ''),
                    "article": getattr(r, 'article_number', ''),
                    "reference": ref,
                    "content": full_content,          # 完整正文，给前端展示用
                })
                if full_content:
                    # 注入 prompt 时每条截断 200 字
                    prompt_contents.append(full_content[:200] + ("..." if len(full_content) > 200 else ""))

            return {
                "reference": "；".join(ref["reference"] for ref in references),
                "references": references,             # 含完整 content
                "content": "\n\n".join(prompt_contents),  # 截断版，用于 prompt
            }
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")

    return {"reference": default_ref, "references": [], "content": ""}

# ==========================================
# 全局异常处理
# ==========================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# ==========================================
# 启动事件
# ==========================================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化 RAG"""
    initialize_rag()

# ==========================================
# API 端点
# ==========================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "rag_available": RAG_AVAILABLE,
        "rag_initialized": retriever is not None if RAG_AVAILABLE else False
    }

@app.get("/api/v1/kb/status")
async def get_kb_status():
    """获取知识库状态"""
    if not RAG_AVAILABLE or legal_kb_loader is None:
        return {"available": False, "message": "RAG 模块不可用"}
    
    try:
        meta = legal_kb_loader.get_coverage_summary()
        return {
            "available": True,
            "version": meta.version,
            "laws_count": meta.laws_count,
            "total_articles": meta.total_articles,
            "violation_types": list(meta.coverage.keys())
        }
    except Exception as e:
        return {"available": False, "message": str(e)}

@app.post("/api/v1/kb/search")
async def search_knowledge(
    query: str,
    top_k: int = 5,
    current_user: User = Depends(get_current_user)
):
    """检索法律知识库"""
    if not RAG_AVAILABLE or retriever is None:
        raise HTTPException(status_code=503, detail="RAG 模块不可用")
    
    try:
        results = retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "text": r.text,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sentences = split_into_sentences(request.text)
    # violation_flags 改为按 violation_id（如 "I1", "I2"）标记，每种违规只扣一次
    violation_flags = {info["id"]: 0 for info in INDICATORS.values()}
    violations_list = []
    # 每条句子的原始分类结果（测试集式明细）
    sentence_results = []

    for idx, sentence in enumerate(sentences):
        # 标题行过滤：跳过标题/小标题，避免被 RoBERTa 误判为违规
        if is_likely_heading(sentence):
            sentence_results.append({
                "index": idx + 1,
                "sentence": sentence,
                "class_name": "[标题已跳过]",
                "max_class_idx": -1,
                "max_prob": 0,
                "confidence": None,
                "raw_probs": [],
                "detected_violations": [],
                "skipped_reason": "heading",
            })
            continue

        # roberta_predict 现在返回完整分类明细字典
        pred = roberta_predict(sentence)

        # 记录逐句原始分类结果
        sentence_results.append({
            "index": idx + 1,
            "sentence": sentence,
            "class_name": pred.get("class_name", ""),
            "max_class_idx": pred.get("max_class_idx", -1),
            "max_prob": pred.get("max_prob", 0),
            "confidence": pred.get("confidence"),
            "raw_probs": pred.get("raw_probs", []),
            "detected_violations": list(pred.get("mapped", {}).keys()),
        })

        # 从 mapped 结果中提取违规（仅保留 prob > 0.5 的）
        # 每个句子的每个违规类型都独立记录，不去重
        for violation_id, prob in pred.get("mapped", {}).items():
            if prob > 0.5:
                violation_flags[violation_id] = 1
                indicator_name = ID_TO_INDICATOR.get(violation_id, "")
                rag_legal = get_legal_basis_from_rag(violation_id, context=sentence)
                violations_list.append({
                    "indicator": indicator_name,
                    "violation_id": violation_id,
                    "snippet": sentence,
                    "legal_basis": rag_legal["reference"],
                    "legal_detail": rag_legal["content"],
                    "legal_references": rag_legal.get("references", []),
                    "confidence": round(prob, 4),
                })

    # violation_flags 现在以 violation_id (如 "I1") 为 key
    penalty = sum(
        INDICATORS[ID_TO_INDICATOR[v_id]]["weight"] * flag
        for v_id, flag in violation_flags.items()
    )
    total_score = round(max(0.0, 100.0 - (penalty * 100.0)), 1)

    if total_score >= SCORE_THRESHOLD_LOW_RISK:
        risk_level = "低风险"
    elif SCORE_THRESHOLD_HIGH_RISK <= total_score < SCORE_THRESHOLD_LOW_RISK:
        risk_level = "中等风险"
    else:
        risk_level = "高风险"

    project_id = f"p{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow()
    project = Project(
        id=project_id,
        user_id=current_user.id,
        name=f"审查-{now.strftime('%Y%m%d-%H%M%S')}",
        source_type=request.source_type,
        score=total_score,
        risk_level=risk_level,
        result_json=json.dumps(violations_list),
        raw_text=request.text[:5000]
    )
    db.add(project)
    db.commit()
    
    return {
        "id": project.id,
        "name": project.name,
        "score": project.score,
        "risk_level": project.risk_level,
        "violations": violations_list,
        "sentence_results": sentence_results,       # 每条句子的原始分类明细
    }

@app.post("/api/v1/rectify")
async def rectify_snippet(
    request: RectifyRequest,
    current_user: User = Depends(get_current_user)
):
    """整改违规条款"""
    # 使用 RAG 检索相关法律条款（返回结构化数据：引用 + 正文）
    rag_legal = get_legal_basis_from_rag(request.violation_type, context=request.original_snippet)
    legal_reference = rag_legal["reference"]       # 用于展示和返回给前端
    legal_content = rag_legal["content"]           # RAG 检索到的法律条文正文（注入 prompt）

    # 如果 RAG 没有返回正文，用前端传入的兜底
    if not legal_content:
        legal_content = request.legal_basis or ""

    # 获取整改提示语（从 violation_config 统一获取）
    violation_hint = ID_TO_HINT.get(request.violation_type, "【重要】必须符合《个人信息保护法》相关要求。")

    # 根据 mode 构建差异化 prompt + chat messages
    if request.mode == "summary":
        # ====== 摘要模式：条款本质 + 风险点拨 ======
        user_content = f"""# Rules
1. **语义简化**：不使用法律术语，而是用普通人日常生活的词汇。
2. **客观陈述**：去掉带有强烈感情色彩的贬义词（如"偷走"、"流氓"），改为客观描述其行为本质。
3. **结构严谨**：严格按照下方格式输出，字数保持精炼。
4. **纯净输出**：禁止输出任何格式标记（如 #、###、>、**、- 之类的 Markdown 符号），只输出纯文本内容。

---

# Output Format（按顺序输出以下两部分，每部分开头用中文标题）

第一部分标题：条款本质
输出一句引用格式的文字，说明条款的实际含义。
示例：这句条款的实际意思是：公司在用户注册时，即使未明确告知用途，也会收集用户的通讯录和位置信息。

第二部分标题：风险点拨
分两段输出，每段开头用中文标签：
实际行为：[一段话解释该条款在现实中如何运作]
潜在影响：[一句话说明可能导致的结果]

---

【原条款】
{request.original_snippet}

【合规风险】
{violation_hint}

【相关法律依据】
{legal_reference}
{legal_content}

请严格按上方 Output Format 输出，只输出两部分正文内容，不要添加任何额外文字、前缀、格式符号或分隔线。"""
        system_content = "你是隐私政策合规解读专家。请严格按用户指定的 Output Format 输出纯净的纯文本内容，不要使用任何 Markdown 格式标记。"
    else:
        # ====== 改写模式：依法精准删改 ======
        # 核心设计原则：
        #   改写必须由 RAG 检索到的【具体法律条文】驱动，而非通用模板。
        #   模型需要知道：违反了哪条法的哪一款 → 该款怎么规定的 → 原文哪里违例 → 怎么改才合规
        #
        # Prompt 结构：原条款 + 法律依据(压缩为约束条件) + 改写指令(强否定约束)
        violation_type_name = ID_TO_INDICATOR.get(request.violation_type, request.violation_type)

        # 直接复用函数开头已检索的 rag_legal（含 references 结构化列表），不重复调用 RAG
        legal_ref_list = rag_legal.get("references", [])

        if legal_ref_list:
            # 有 RAG 结果：将法律条文提炼为「约束指令」而非大段引用
            # 关键：不把法律正文塞给模型，而是告诉模型「法律要求什么」
            law_constraints = []
            for ref in legal_ref_list:
                ref_content = ref.get("content", "").strip()
                if ref_content:
                    # 只取前 80 字，且加上「要求：」前缀，变成约束而非引文
                    truncated = ref_content[:80] + ("..." if len(ref_content) > 80 else "")
                    law_constraints.append(
                        f"- 要求：{truncated}（来源：《{ref['law']}》{ref['article']}）"
                    )
            legal_basis_detail = "\n".join(law_constraints)
        else:
            # RAG 无结果时降级为静态配置
            static_ref = INDICATORS.get(violation_type_name, {}).get("legal_basis", "")
            static_hint = ID_TO_HINT.get(request.violation_type, "")
            legal_basis_detail = f"- 要求：{static_ref}\n  合规要点：{static_hint}"

        user_content = f"""【原条款】
{request.original_snippet}

【合规要求（法律规定的硬性约束，你必须满足这些要求）】
{legal_basis_detail}

【改写规则】
1. 直接输出修改后的隐私政策正文，不要任何前缀、标题、编号、解释
2. 删除违反上述要求的表述，保留不违规的部分
3. 如果某项信息收集缺乏法律要求的必要说明，用最短的方式补上（不超过半句）
4. 输出总字数严格控制在原文的 70% 以内

【禁止事项 — 违反以下任一条即为失败】
- 禁止输出「根据XX法」「依照XX规定」「依据XX法第X条」等引用语
- 禁止复述、概括、转述上述法律条文的内容
- 禁止输出「修改说明」「改动理由」「对比分析」等解释性文字
- 禁止输出 Markdown 格式标记（# * - > 等）
- 禁止输出原标题或标注「改写后」「修订版」等标签
- 禁止在正文之外输出任何其他内容"""

        system_content = "你是一个只做不改说的文案编辑。你的唯一任务是：输入一段有法律问题的隐私政策条款，输出修改后的干净正文。像删除键一样工作——删掉问题的，保留没问题的。绝对不要解释、不要引用法律、不要加注释。只给结果。"

    # 调用 LLM 生成整改建议（兼容三种模式：github / local / hf）
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    if LLM_MODE == "github" and llm_github is not None:
        # GitHub Models 模式（推荐）：OpenAI 兼容接口，免费 Phi-4
        response = llm_github.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        suggested_text = response.choices[0].message.content.strip()
    elif LLM_MODE == "local" and hasattr(llm, 'create_chat_completion'):
        # 本地 GGUF 模式
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        suggested_text = response["choices"][0]["message"]["content"].strip()
    else:
        # HF Inference API 模式（fallback）
        response = llm.chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        suggested_text = response.choices[0].message.content.strip()
    
    return {
        "suggested_text": suggested_text,
        "legal_basis": legal_reference,   # 返回引用格式（如"《个人信息保护法》第28条"）给前端展示
        "legal_detail": legal_content,     # 返回完整条文内容（前端可选择性展示）
        "mode": request.mode,
    }

@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    return {"text": text}

@app.post("/api/v1/fetch-url")
async def fetch_url(
    request: UrlRequest,
    current_user: User = Depends(get_current_user)
):
    import requests
    from bs4 import BeautifulSoup
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取URL内容: {str(e)}")

@app.get("/api/v1/projects")
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.user_id == current_user.id).order_by(Project.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "score": p.score,
            "risk_level": p.risk_level,
            "created_at": p.created_at.isoformat()
        }
        for p in projects
    ]

@app.get("/api/v1/projects/{project_id}")
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    return {
        "id": project.id,
        "name": project.name,
        "score": project.score,
        "risk_level": project.risk_level,
        "violations": json.loads(project.result_json) if project.result_json else [],
        "created_at": project.created_at.isoformat()
    }

class UpdateProjectRequest(BaseModel):
    violations: List[dict]

@app.put("/api/v1/projects/{project_id}")
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新项目的违规条款数据（如采纳整改建议后回写 suggested_text）"""
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    project.result_json = json.dumps(request.violations, ensure_ascii=False)
    db.commit()

    return {"message": "更新成功", "id": project.id}

@app.get("/api/v1/export/{project_id}")
async def export_report(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    violations = json.loads(project.result_json) if project.result_json else []
    
    report = f"""隐私政策合规审查报告
==================

项目名称：{project.name}
审查时间：{project.created_at.strftime('%Y-%m-%d %H:%M')}
合规得分：{project.score}
风险等级：{project.risk_level}

违规条款统计
-----------
共发现 {len(violations)} 项潜在风险

详细分析
-------
"""
    for i, v in enumerate(violations, 1):
        report += f"\n{i}. {v.get('indicator', '未知类别')} (ID: {v.get('violation_id', 'N/A')})\n"
        report += f"   原文：{v.get('snippet', '未知')}\n"
        report += f"   依据：{v.get('legal_basis', '未知')}\n"
        suggested = v.get('suggested_text', '')
        if suggested:
            report += f"   整改建议：{suggested}\n"
        reason = v.get('reason', '')
        if reason and reason != v.get('indicator', ''):
            report += f"   说明：{reason}\n"
    
    return Response(
        content=report,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=report_{project_id}.txt"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
