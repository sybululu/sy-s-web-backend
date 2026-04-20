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
    """加载微调 checkpoint 到自定义模型结构（11类多分类）"""
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
        logger.warning(f"RoBERTa 多分类 缺失键: {missing}")
    if unexpected:
        logger.warning(f"RoBERTa 多分类 多余键: {unexpected}")

    model.eval()
    return model


# ─── 二分类模型：判断句子是否存在违规风险（有违规=1 / 无违规=0） ──
# 训练代码结构与多分类完全相同，仅 num_classes=2（输出维度不同）
class CustomBertBinaryModel(nn.Module):
    """二分类模型：BertModel + fc(768→2)，用于前置过滤是否违规"""

    def __init__(self):
        super(CustomBertBinaryModel, self).__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # 二分类：num_classes=2，类别 0=无违规风险, 1=有违规风险
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch, 768]
        return self.fc(pooled_output)          # [batch, 2]


def load_binary_model(ckpt_path: str) -> CustomBertBinaryModel:
    """加载二分类 checkpoint"""
    model = CustomBertBinaryModel()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    cleaned_state_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    if missing:
        logger.warning(f"RoBERTa 二分类 缺失键: {missing}")
    if unexpected:
        logger.warning(f"RoBERTa 二分类 多余键: {unexpected}")

    model.eval()
    return model


# 初始化 tokenizer 和自定义模型
tokenizer_roberta = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 1a. 加载 11 类多分类模型（第二阶段：判定违规类型）
roberta_ckpt_path = hf_hub_download(
    repo_id="sybululu/bert-moe",
    filename="multi_classification_bertmoe.ckpt"
)
model_roberta = load_trained_model(roberta_ckpt_path)
print("✅ RoBERTa 多分类模型加载完成（11类 → 12类违规类型映射）")

# 1b. 加载二分类模型（第一阶段：判定是否有违规风险）
binary_ckpt_path = hf_hub_download(
    repo_id="sybululu/bert-moe",
    filename="risk_identification_bertmoe.ckpt",
    revision="06ef3d5e733870c99ee763eec552aea1d1a3f709"
)
model_binary = load_binary_model(binary_ckpt_path)
print("✅ RoBERTa 二分类模型加载完成（有/无违规风险 前置过滤）")

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
# 定义为模块级常量，避免每次调用 is_likely_heading() 时重复编译
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

# 强标题正则（更严格的匹配，用于 is_likely_heading 内部二次判断）
_STRONG_HEADING = re.compile(
    r'^('
    # 中文序号: 一、/ 二、/ （一）/ 1、/ 1．(注意中文顿号和句点)
    r'[一二三四五六七八九十\d]+[\.\、\．]'
    # 带章节后缀: 第一章 / 1.1节 / 第X条 / 第X节 / 第X款 / 第X部分
    r'|第[一二三四五六七八九十\d]+[\.\、\．\s]*(?:节|章|条款|部分|款|条)'
    r'|[一二三四五六七八九十\d]+[\.\、\．\s]*(?:节|章|条款|部分|款|条)'
    # 全角括号序号: （一）/ （1）
    r'|[（][一二三四五六七八九十\d][）]'
    # 数字+顿号/点: "1、" / "1．" / "1. "
    r'|\d+[\.\、\．]\s*'
    # 常见隐私政策标题关键词白名单
    r'|(?:信息收集|信息使用|信息共享|数据安全|用户权利|Cookie|未成年人|'
    r'联系我们|政策更新|生效时间|适用范围|定义|总则|附则|修订记录'
    r'|您提供的信息|我们收集的信息|我们如何使用|我们如何共享'
    r'|第三方服务|数据留存|安全措施|未成年人保护|您的权利'
    r')'
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



# ═══ 关键词熔断词表：命中则无视二分类结果，强制判定为违规 ═══
# 覆盖：强制授权、模糊收集、超范围共享、无期限留存等典型流氓表述
CIRCUIT_BREAKER_KEYWORDS = [
    # 强制授权 / 默认同意
    "默认勾选", "默认同意", "视为同意", "即表示同意", "注册即同意",
    "使用本服务即视为", "一经注册", "自动视为",
    # 超范围收集敏感数据
    "生物识别", "指纹", "人脸识别", "虹膜", "步态", "基因", "DNA",
    "健康数据", "医疗记录", "性生活", "性取向",
    "宗教信仰", "政治倾向", "种族", "民族起源",
    "通讯录", "联系人", "短信内容", "通话记录", "位置轨迹", "定位信息",
    # 模糊目的收集
    "提升体验", "优化服务", "改善产品", "业务需要", "必要的",
    "为了更好地", "为您提供更优", "可能收集", "包括但不限于",
    # 第三方共享无明确范围
    "向第三方共享", "提供给合作伙伴", "允许第三方", "可能会分享",
    "转交", "转让给", "披露给",
    # 无明确期限
    "永久保存", "无限期", "直至您删除", "在必要期间内保留",
    "我们根据需要", "合理期限内",
]

# 二分类阈值配置
BINARY_THRESHOLD_STRICT = 0.35    # 宁可错杀：risk_prob > 0.35 即判定为违规
BINARY_FUZZY_LOW = 0.15           # 模糊地带下界：0.15 <= risk <= 0.35 → 送Phi-4二审


def keyword_circuit_breaker(sentence: str) -> Optional[str]:
    """
    关键词熔断：扫描句子是否包含极其流氓的词汇。
    
    返回命中的关键词（如有），None 表示未命中。
    命中时无论模型概率多低，都应强行判定为违规。
    """
    for kw in CIRCUIT_BREAKER_KEYWORDS:
        if kw in sentence:
            return kw
    return None


async def llm_judge_violation(sentence: str) -> bool:
    """
    LLM as Judge（二审法官）：对模糊地带句子用 Phi-4 判断是否违规。
    
    仅用于二分类模型不确定的边缘案例，不替代主流路径。
    """
    prompt = (
        "你是一个隐私政策合规审查专家。请严格判断以下句子是否包含违规逻辑。\n"
        '只回答"是"或"否"，不要解释。\n\n'
        f"句子：{sentence}"
    )
    try:
        if llm_github:
            resp = llm_github.chat.completions.create(
                model=LLM_MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是隐私政策合规审查专家，只回答'是'或'否'。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8,
                temperature=0.1,
            )
            answer = resp.choices[0].message.content.strip()
            result = "是" in answer or "违规" in answer or "yes" in answer.lower()
            logger.info(f"[LLM二审] 句子: {sentence[:30]}... → {'⚠️违规' if result else '✅安全'} (原始回答: {answer})")
            return result
        elif llm:
            resp = llm.chat_completion(
                messages=[
                    {"role": "system", "content": "你是隐私政策合规审查专家，只回答'是'或'否'。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=8,
                temperature=0.1,
            )
            answer = str(resp).strip()
            result = "是" in answer or "违规" in answer
            logger.info(f"[LLM二审-HF] 句子: {sentence[:30]}... → {'⚠️违规' if result else '✅安全'}")
            return result
    except Exception as e:
        logger.warning(f"[LLM二审] 调用失败，保守判定为违规: {e}")
        return True  # 保守策略：LLM调用失败时不放过

    # 无可用LLM时保守判定为违规
    return True


def binary_predict(sentence: str) -> Dict[str, Any]:
    """
    二分类前置过滤（增强版）：三重判断机制
    
    判定优先级（从高到低）：
      1. 关键词熔断 → 直接违规（无视模型）
      2. 模型高置信 → risk_prob > 0.35 直接违规
      3. 模型低置信 → risk_prob < 0.15 直接安全
      4. 模糊地带 → 0.15 <= risk_prob <= 0.35，标记送Phi-4二审
    
    返回:
    {
        "is_violation": bool,       # 最终判定
        "risk_prob": float,         # 类别1(正例/有违规)的概率
        "safe_prob": float,         # 类别0(余例/无违规)的概率
        "logits": [float, float],
        "triggered_by": str,        # "keyword"/"model_high"/"model_low"/"fuzzy_llm"/"llm_safe"
        "matched_keyword": str|None,# 熔断命中的关键词
        "needs_llm_judge": bool,    # 是否需要Phi-4二审
    }
    """
    # ── 第0关：关键词熔断（最高优先级）──
    matched_kw = keyword_circuit_breaker(sentence)
    if matched_kw:
        logger.info(
            f"[二分类-熔断] 句子: {sentence[:40]}{'...' if len(sentence)>40 else ''} "
            f"| 命中关键词: '{matched_kw}' → ⚠️强制违规"
        )
        return {
            "is_violation": True,
            "risk_prob": 1.0,
            "safe_prob": 0.0,
            "logits": [0.0, 999.0],
            "triggered_by": "keyword",
            "matched_keyword": matched_kw,
            "needs_llm_judge": False,
        }

    # ── 第1关：二分类模型推理 ──
    inputs = tokenizer_roberta(sentence, return_tensors="pt", truncation=True, max_length=150)
    with torch.no_grad():
        outputs = model_binary(**inputs)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.squeeze()
        else:
            logits = outputs.logits.squeeze()

        probs = torch.softmax(logits, dim=-1).tolist()

    if not isinstance(probs, list):
        probs = [probs]

    safe_prob = probs[0]   # 类别0: 余例/无违规风险
    risk_prob = probs[1]   # 类别1: 正例/有违规风险

    # ── 第2关：三段式判定 ──
    if risk_prob > BINARY_THRESHOLD_STRICT:
        # 高置信违规区
        logger.info(
            f"[二分类-模型] 句子: {sentence[:40]}{'...' if len(sentence)>40 else ''} "
            f"| 有违规={risk_prob:.4f} > 阈值{BINARY_THRESHOLD_STRICT} → ⚠️违规"
        )
        return {
            "is_violation": True,
            "risk_prob": round(risk_prob, 6),
            "safe_prob": round(safe_prob, 6),
            "logits": [round(l, 4) for l in (logits.tolist() if isinstance(logits, torch.Tensor) else logits)],
            "triggered_by": "model_high",
            "matched_keyword": None,
            "needs_llm_judge": False,
        }

    if risk_prob < BINARY_FUZZY_LOW:
        # 高置信安全区
        logger.info(
            f"[二分类-模型] 句子: {sentence[:40]}{'...' if len(sentence)>40 else ''} "
            f"| 有违规={risk_prob:.4f} < 下界{BINARY_FUZZY_LOW} → ✅安全"
        )
        return {
            "is_violation": False,
            "risk_prob": round(risk_prob, 6),
            "safe_prob": round(safe_prob, 6),
            "logits": [round(l, 4) for l in (logits.tolist() if isinstance(logits, torch.Tensor) else logits)],
            "triggered_by": "model_low",
            "matched_keyword": None,
            "needs_llm_judge": False,
        }

    # 模糊地带：标记需要Phi-4二审（由 analyze 循环异步调用）
    logger.info(
        f"[二分类-模糊] 句子: {sentence[:40]}{'...' if len(sentence)>40 else ''} "
        f"| 有违规={risk_prob:.4f} ∈ 模糊带[{BINARY_FUZZY_LOW}, {BINARY_THRESHOLD_STRICT}] → ⏳待LLM二审"
    )
    return {
        "is_violation": False,          # 先标记为安全，等LLM结果后覆盖
        "risk_prob": round(risk_prob, 6),
        "safe_prob": round(safe_prob, 6),
        "logits": [round(l, 4) for l in (logits.tolist() if isinstance(logits, torch.Tensor) else logits)],
        "triggered_by": "fuzzy_llm",
        "matched_keyword": None,
        "needs_llm_judge": True,         # 标记需要Phi-4二审
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


def format_legal_paired(rag_legal: Dict) -> str:
    """
    将 RAG 检索结果格式化为「引用+正文」一一配对的展示文本。

    输入: get_legal_basis_from_rag() 的返回值
    输出: 每条法律以「《法名》第X款\\n正文内容」成对排列的文本
    """
    if not rag_legal:
        return ""
    if not rag_legal.get("references"):
        # 无 references 时降级为 reference + content 拼接
        ref = rag_legal.get("reference", "")
        content = rag_legal.get("content", "")
        return f"{ref}\n{content}" if content else ref

    paired_lines = []
    for ref in rag_legal["references"]:
        ref_title = f"《{ref['law']}》{ref['article']}"
        ref_content = ref.get("content", "").strip()
        if ref_content:
            paired_lines.append(f"{ref_title}\n{ref_content}")
        else:
            paired_lines.append(ref_title)
    return "\n\n".join(paired_lines)

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

        # ═══ 第一阶段：二分类前置过滤（三重判断）═══
        binary_result = binary_predict(sentence)
        triggered_by = binary_result.get("triggered_by", "unknown")

        # ── 模糊地带：Phi-4 二审法官 ──
        if binary_result.get("needs_llm_judge"):
            llm_verdict = await llm_judge_violation(sentence)
            triggered_by = "llm_judge" if llm_verdict else "llm_safe"
            
            if not llm_verdict:
                # LLM二审判定安全 → 跳过
                sentence_results.append({
                    "index": idx + 1,
                    "sentence": sentence,
                    "class_name": "[安全-LLM]",
                    "max_class_idx": -1,
                    "max_prob": 0,
                    "confidence": None,
                    "raw_probs": [],
                    "detected_violations": [],
                    "skipped_reason": "llm_safe",
                    "binary_risk_prob": binary_result["risk_prob"],
                    "binary_safe_prob": binary_result["safe_prob"],
                    "triggered_by": triggered_by,
                })
                continue
            # LLM二审判定违规 → 继续进入第二阶段多分类

        if not binary_result["is_violation"] and not binary_result.get("needs_llm_judge"):
            # 确实安全 → 跳过
            skip_reason = {
                "keyword": "熔断关键词" if binary_result.get("matched_keyword") else None,
                "model_low": "模型高置信安全",
            }.get(triggered_by, "binary_safe")
            
            sentence_results.append({
                "index": idx + 1,
                "sentence": sentence,
                "class_name": f"[{skip_reason}]",
                "max_class_idx": -1,
                "max_prob": 0,
                "confidence": None,
                "raw_probs": [],
                "detected_violations": [],
                "skipped_reason": skip_reason,
                "binary_risk_prob": binary_result["risk_prob"],
                "binary_safe_prob": binary_result["safe_prob"],
                "triggered_by": triggered_by,
                "matched_keyword": binary_result.get("matched_keyword"),
            })
            continue

        # ═══ 第二阶段：11类多分类 → 12类违规映射 ═══
        pred = roberta_predict(sentence)

        # 记录逐句原始分类结果（含二分类信息）
        sentence_results.append({
            "index": idx + 1,
            "sentence": sentence,
            "class_name": pred.get("class_name", ""),
            "max_class_idx": pred.get("max_class_idx", -1),
            "max_prob": pred.get("max_prob", 0),
            "confidence": pred.get("confidence"),
            "raw_probs": pred.get("raw_probs", []),
            "detected_violations": list(pred.get("mapped", {}).keys()),
            "binary_risk_prob": binary_result["risk_prob"],
            "binary_safe_prob": binary_result["safe_prob"],
            "triggered_by": triggered_by,
        })

        # 从 mapped 结果中提取违规（问题4: 不再需要prob>0.5过滤，直接取映射结果）
        # 问题5: 按 (snippet, violation_id) 去重，避免同一句话重复同一标签
        seen_violations_for_this_sentence = set()
        for violation_id, prob in pred.get("mapped", {}).items():
            dedup_key = (sentence, violation_id)
            if dedup_key in seen_violations_for_this_sentence:
                continue
            seen_violations_for_this_sentence.add(dedup_key)
            
            violation_flags[violation_id] = 1
            indicator_name = ID_TO_INDICATOR.get(violation_id, "")
            rag_legal = get_legal_basis_from_rag(violation_id, context=sentence)
            violations_list.append({
                "indicator": indicator_name,
                "violation_id": violation_id,
                "snippet": sentence,
                "legal_basis": format_legal_paired(rag_legal),
                "legal_detail": "",
                "legal_references": rag_legal.get("references", []),
                "confidence": round(prob, 4),
                "triggered_by": triggered_by,
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

    # ── 从 RAG 提取法律引用（仅引用名，不含正文 —— 给权威锚点但不给复读素材） ──
    rag_refs = rag_legal.get("references", [])
    if rag_refs:
        legal_citations = "\n".join(f"  - 《{r['law']}》{r['article']}" for r in rag_refs)
    else:
        legal_citations = f"  - {rag_legal.get('reference', '《个人信息保护法')}"

    # 根据 mode 构建差异化 prompt + chat messages
    if request.mode == "summary":
        # ====== 摘要模式：基于法律依据的专业解读 ======
        #
        # 策略：
        #   1. 给法律引用名（"《个保法》第六条"）作为权威锚点，不给正文 → 不复读
        #   2. 要求模型"基于以上法律分析"，用自身知识解读 → 有专业性
        #   3. 固定四段式结构 → 逻辑清晰不散
        #   4. 明确禁止逐字引用条文 → 避免复读

        VIOLATION_CORE_ISSUE = {
            "I1": "收集敏感个人信息（生物识别、健康、位置、通讯录等），但未说明收集目的或超出必要范围",
            "I2": "收集用户信息但未说明具体用途，使用'提升体验'等模糊表述",
            "I3": "将同意与服务捆绑，用户不同意则无法使用（默认勾选/即视为同意）",
            "I4": "收集的信息范围超出了提供服务实际需要",
            "I5": "提及向第三方共享信息，但未说明共享对象、范围或种类",
            "I6": "向第三方提供信息前未取得用户单独同意",
            "I7": "允许第三方使用数据，但未限定具体目的和范围",
            "I8": "未说明数据的留存期限",
            "I9": "未说明留存期满后的数据销毁或匿名化处理方式",
            "I10": "未完整告知用户对其数据享有的法定权利",
            "I11": "提到用户权利但未提供便捷的行使途径（在线入口/联系方式）",
            "I12": "未承诺权利请求的响应时限",
        }
        core_issue = VIOLATION_CORE_ISSUE.get(request.violation_type,
            "存在合规风险，可能损害用户权益")

        user_content = f"""请对以下隐私政策条款进行合规风险分析。

【条款原文】
{request.original_snippet}

【违规类型】{core_issue}

【相关法律依据】
{legal_citations}

请严格按以下四个部分回答，每部分用标题开头：

一、问题诊断
指出这条条款违反了上面列出的哪条（或哪几条）法律的具体要求，用一两句话说明违在哪里。（提到法律时只用法名+条目号，不要引用原文）

二、通俗解读
用大白话翻译这条条款——它实际上在干什么、对用户意味着什么。不要复述原文，用自己的话重新表述。

三、影响评估
分两点，每点一句话：
• 日常影响：普通用户会因此遭遇什么
• 最坏情况：如果被滥用可能造成什么后果

四、整改方向
一句话说明怎么改才能符合上述法律要求。"""

        system_content = ("你是隐私合规顾问，擅长将法律要求转化为普通人能理解的分析。"
            "你的回答要有法律依据支撑（引用具体法条），但表达要自然流畅，像在给同事做简报。"
            "结构清晰、逻辑连贯、不啰嗦。绝对不要大段引用或复述法律条文原文。")
    else:
        # ====== 改写模式：依法精准删改（零法律文本输出） ======
        #
        # 核心策略：
        #   1. 法律知识内化为编辑指令（怎么改），不外传给模型任何法律文本
        #   2. system + user 双层强制禁令：禁止输出任何法条名/条文/法律术语
        #   3. 给 before→after 示例锁定风格：只输出干净的用户友好文案
        #   4. legal_citations 仅用于内部选择 edit_instruction，不注入 prompt

        EDIT_INSTRUCTIONS = {
            "I1": "【必须执行的操作】(1)删除身份证件号码、银行信息——绝大多数服务不需要这些敏感信息；(2)性别、年龄、联系人信息也删除，除非你能说出具体哪个功能非用它不可；(3)保留的每项信息后面必须加'用于[具体功能]'说明；(4)开头加'我们仅在以下场景收集必要信息：'，然后用编号列表逐条列出。最终输出应该比原句短一半以上。",
            "I2": "【必须执行的操作】(1)找到每项信息收集，在后面补上具体用途，格式为'——用于[某功能]'；(2)删除'提升体验''优化服务''业务需要'等所有万能借口，替换为真实场景名称（如'订单配送需地址''客服回拨需电话'）；(3)把'可能收集''将收集'等模糊措辞改为确定的'仅在[某操作]时收集'。",
            "I3": "【必须执行的操作】(1)删除'即视为同意''注册即表示''使用本服务即视为'等所有捆绑同意表述；(2)改为两步流程：先说'请您审阅本政策后主动勾选确认框'，再说'该同意可随时撤回'；(3)不能有任何默认勾选或隐含同意的暗示。",
            "I4": "【必须执行的操作】(1)逐项审查原句列出的每个信息类型，问自己'这个功能真的需要它吗？'，不需要的直接删掉；(2)用'仅限以下场景：'开头，然后编号列表列出保留项，每项格式为'[信息类型]——[唯一对应的功能场景]'；(3)原句中笼统的'等信息''等相关'必须替换为具体列举或直接删除。",
            "I5": "【必须执行的操作】(1)'第三方''合作伙伴'等模糊称谓必须替换为至少2个具体类型（如'支付服务商、物流配送方'），如果原文没给具体信息就写'支付服务商、物流配送方、数据分析服务商'作为合理示例；(2)列出共享的具体数据种类（不要说'所有个人数据'）；(3)列出共享目的。格式改为'我们可能向以下类型的第三方提供特定信息：[类型]——用于[目的]，涉及的数据包括[具体种类]。",
            "I6": "【必须执行的操作】(1)在第三方共享描述前强制加上'我们将单独征得您的明确同意后'；(2)加一句'此独立于本政策的其他同意条款'强调单独性；(3)删除任何暗示一次性同意即可覆盖所有场景的表述。",
            "I7": "【必须执行的操作】(1)在共享描述后加上接收方的用途限制条款；(2)标准措辞：'接收方仅可将数据用于上述指定用途，不得转售、不得用于其自身营销或其他未获授权之目的'；(3)如果原句完全没有这个限制，必须补上。",
            "I8": "【必须执行的操作】(1)为每类数据分别添加留存时限，禁止使用'必要期间''合理期限''我们不再需要时'等无期限表述；(2)标准格式：'[某类信息]保存期限为[时间]，届满后自动删除/匿名化'；(3)如果没有具体业务背景，统一写'保存期限为 achieving 目的后6个月，届时自动匿名化处理'。",
            "I9": "【必须执行的操作】(1)补充具体销毁方式，不能只说'我们会删除'；(2)标准措辞：'保存期限届满后，我们将通过安全擦除技术使数据无法恢复'；(3)如果原句没有销毁说明，整句补上。",
            "I10": "【必须执行的操作】(1)完整列举5项核心权利：查询、更正、删除、撤回同意、数据可携带；(2)每项后面加操作入口，如'(设置→账户管理→个人信息)'；(3)用一句话概括投诉渠道。格式为编号列表。",
            "I11": "【必须执行的操作】(1)提供至少两种联系方式：应用内路径 + 外部渠道；(2)应用内路径要具体到菜单层级，如'设置→隐私→意见反馈'；(3)外部渠道给邮箱或电话；(4)加时限承诺'15个工作日内响应'。",
            "I12": "【必须执行的操作】(1)删除'尽快''及时''合理期限内'等无时限表述；(2)替换为'15个工作日内完成处理并以您提供的联系方式反馈结果'；(3)如果原句没有时限，整句补上。",
        }
        edit_instruction = EDIT_INSTRUCTIONS.get(request.violation_type,
            "修改条款使其符合合规要求：删除模糊表述、缩小收集范围、增加用户选择权、明确期限和联系方式。")

        user_content = f"""你需要对以下隐私政策条款进行合规改写。这不是润色，是强制性改写——原条款存在合规缺陷，你必须修复它。

【原条款（有问题的原文）】
{request.original_snippet}

【法律依据 — 阅读后理解合规要求，不可出现在输出中】
{legal_content}

【本次改写必须执行的操作】
{edit_instruction}

【输出要求】
1. 直接输出改写后的条款正文，不要任何前缀、解释、标注
2. 必须对原条款做实质性修改——删除违规措辞、补充缺失要素、缩小过度范围。如果只是缩句或换说法而不解决合规问题，就是不合格的改写
3. 用最少的字把该说的都说清楚，没有一个废字
4. 输出中禁止出现任何法律名称、条文编号、条文原文、"根据XX法"等引用性表述

【改写范例 —— 注意范例都是大刀阔斧的结构性修改，不是缩句】

范例1：
原句：「为了提供更好的服务，我们可能会收集您的位置信息、通讯录等。」
改写：「我们仅在用户主动使用特定功能时收集必要信息：地图导航需位置信息，消息功能需通讯录权限。用户可随时在「设置-隐私」中撤销授权。」
（改动：加了功能限定+逐项列明用途+用户控制权）

范例2：
原句：「注册即视为您已同意本政策全部条款。」
改写：「完成注册前请审阅本政策并勾选确认框。您可随时在账户设置中撤回同意，不影响此前处理的合法性。」
（改动：捆绑同意→主动确认+撤回权）

范例3：
原句：「我们将向第三方共享所有您的个人数据」
改写：「经您单独明确同意后，我们可能向以下第三方提供必要信息：(1)支付服务商——用于处理交易付款，涉及支付账号和金额；(2)物流配送方——用于订单配送，涉及收货地址和联系方式。接收方仅可将数据用于上述指定用途，不得转售或用于自身营销。」
（改动：模糊第三方→具体类型+单独同意+用途限制）

现在请对上面的原条款进行改写："""

        system_content = ("你是资深隐私政策合规专家。你的任务是对有缺陷的隐私政策条款进行强制性合规改写。"
            "你会收到原条款、相关法律条文和具体的改写操作指令。请严格按指令执行改写——该删的删、该补的补、该缩小的范围要缩小。"
            "改写标准：字少、事全、无废话、无法律引用。每句话必须有实质合规含义。"
            "绝对禁止在输出中出现任何法律名称、条文编号、条文原文或引用性表述。只输出改写后的条款正文。")

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
        "legal_basis": format_legal_paired(rag_legal),  # 配对格式：引用+正文一一对应
        "legal_detail": "",                            # 已合并到 legal_basis
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        response = requests.get(request.url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        # 编码检测（四层降级策略）：
        # 1. 响应头 Content-Type 声明的 charset
        # 2. requests 库自动检测的 encoding（排除 HTTP 默认的 iso-8859-1）
        # 3. chardet/apparent_encoding 启发式检测
        # 4. utf-8 兜底
        content_type = response.headers.get('Content-Type', '')
        enc = None

        if 'charset=' in content_type.lower():
            raw_enc = content_type.split('charset=')[-1].strip().lower()
            # 排除 HTTP 默认值（iso-8859-1 / us-ascii），它们几乎总是错的
            if raw_enc not in ('iso-8859-1', 'us-ascii', 'ascii'):
                enc = raw_enc

        if not enc and response.encoding and response.encoding.lower() not in ('iso-8859-1', 'us-ascii', 'ascii'):
            enc = response.encoding

        if not enc and response.apparent_encoding:
            enc = response.apparent_encoding

        if not enc:
            enc = 'utf-8'

        try:
            text = response.content.decode(enc, errors='replace')
        except Exception:
            text = response.text

        soup = BeautifulSoup(text, 'html.parser')

        # 移除噪声标签：脚本、样式、导航、页脚等非正文内容
        for tag in soup(['script', 'style', 'noscript', 'svg', 'nav', 'header', 'footer',
                         'iframe', 'aside', 'form', 'button']):
            tag.decompose()

        raw_text = soup.get_text(separator='\n', strip=True)
        # 清理多余空白行
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # SPA 空壳检测：文本过短且包含 JS 提示语，返回明确错误
        spa_keywords = ['javascript enabled', 'enable javascript', '请启用 javascript']
        is_spa_shell = (
            len(text) < 500
            and any(kw in text.lower() for kw in spa_keywords)
        )
        if is_spa_shell:
            raise HTTPException(
                status_code=422,
                detail="该页面需要 JavaScript 渲染（SPA），无法通过静态抓取获取内容。"
                        "建议直接粘贴文本或上传文件。"
            )

        return {"text": text}
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="请求超时（>15秒），请检查 URL 是否可访问或网络是否通畅。")
    except requests.exceptions.SSLError as e:
        raise HTTPException(status_code=422, detail=f"SSL 证书验证失败: {str(e)[:100]}")
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"无法连接到目标服务器: {str(e)[:100]}")
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
