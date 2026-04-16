"""
隐私政策合规审查 API
=====================
后端服务 - HuggingFace Spaces 部署版本

功能:
- 隐私政策文本分析 (12类违规检测)
- RAG 法律知识库检索
- 整改建议生成
- 用户认证与项目管理
"""
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from sqlalchemy.orm import Session

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MT5ForConditionalGeneration, AutoConfig
from diff_match_patch import diff_match_patch

from models import User, Project, get_db, init_db
from auth import router as auth_router, get_current_user

# ==========================================
# 法律条文核心内容提取函数
# ==========================================
def get_legal_core(text: str, max_chars: int = 80) -> str:
    """
    从法律条文中提取核心内容（语义压缩策略）
    寻找"应当"、"不得"、"必须"等核心动词，截取其后的关键义务语句
    """
    if not text:
        return ""
    
    # 法律条文中的核心动词（引导核心义务的词）
    core_signals = ["应当", "不得", "必须", "要求", "严禁", "可以", "有权"]
    
    # 寻找第一个出现的动词位置
    start_idx = 0
    for signal in core_signals:
        pos = text.find(signal)
        if pos != -1:
            start_idx = pos
            break
    
    # 从动词开始截取一定长度，保证语义连贯
    core_content = text[start_idx:start_idx + max_chars]
    
    # 如果截断了，补上省略号
    if len(text) > start_idx + max_chars:
        core_content += "..."
    
    return core_content

# ==========================================
# mT5 安全解码函数
# ==========================================
_tokenizer_gen_vocab_size = None

def safe_decode_mt5(tokenizer, token_ids, skip_special_tokens=True):
    """安全解码 mT5 生成的 token ID，过滤超出 tokenizer 词汇表的 ID"""
    global _tokenizer_gen_vocab_size
    if _tokenizer_gen_vocab_size is None:
        _tokenizer_gen_vocab_size = len(tokenizer)
        logger.info(f"mT5 tokenizer vocab_size: {_tokenizer_gen_vocab_size}")
    
    # 过滤掉超出词汇表的 token ID
    filtered_ids = [tid for tid in token_ids if tid < _tokenizer_gen_vocab_size]
    
    if not filtered_ids:
        return ""
    
    return tokenizer.decode(filtered_ids, skip_special_tokens=skip_special_tokens)

# ==========================================
# 配置日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 导入 RAG 模块
# ==========================================
try:
    from src.loader import LegalKBLoader
    from src.store import VectorStore
    from src.search import Retriever
    from src.config import get_config
    from src.mapper import map_to_12_classes
    RAG_AVAILABLE = True
    logger.info("RAG 模块加载成功")
except ImportError as e:
    logger.warning(f"RAG 模块加载失败: {e}")
    RAG_AVAILABLE = False
    map_to_12_classes = None

# ==========================================
# HuggingFace Spaces 自动配置
# ==========================================
# HF_TOKEN 可选（public 模型不需要，private 才需要）
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PORT = int(os.environ.get("PORT", 7860))
REPO_ID = "sybululu/bert-moe"

# ==========================================
# 模型加载状态
# ==========================================
class ModelStatus:
    """模型加载状态"""
    classifier_loaded = False
    generator_loaded = False
    tokenizer_classifier = None
    model_classifier = None
    tokenizer_generator = None
    model_generator = None

model_status = ModelStatus()

def load_models():
    """加载 HuggingFace 模型"""
    global model_status
    
    logger.info("=" * 50)
    logger.info("开始加载模型...")
    logger.info("=" * 50)
    
    # 1. 登录 HuggingFace (如果有Token)
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            logger.info("HuggingFace 登录成功")
        except Exception as e:
            logger.warning(f"HuggingFace 登录失败: {e}")
    else:
        logger.warning("未设置 HF_TOKEN，将以匿名方式下载模型（可能受限）")
    
    # 2. 加载分类模型 (BERT-MoE, 11类)
    # checkpoint 包含完整权重，直接加载即可
    logger.info("-" * 30)
    logger.info("步骤1/2: 加载分类模型...")
    try:
        from huggingface_hub import hf_hub_download
        
        # 下载包含完整权重的 checkpoint
        cls_ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="multi_classification_bertmoe.ckpt",
            token=HF_TOKEN or None
        )
        logger.info(f"分类模型 checkpoint 已下载: {cls_ckpt_path}")
        
        # 加载 checkpoint (包含完整权重和结构)
        # 注意: PL checkpoint 包含 Lightning 结构，必须 weights_only=False
        checkpoint = torch.load(cls_ckpt_path, map_location="cpu", weights_only=False)
        
        # 处理 checkpoint 格式
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 不再清理键名前缀，但需要处理分类头命名差异
            # Checkpoint 使用 fc.weight，模型期望 classifier.weight
            for key in list(state_dict.keys()):
                if key == 'fc.weight':
                    state_dict['classifier.weight'] = state_dict.pop('fc.weight')
                elif key == 'fc.bias':
                    state_dict['classifier.bias'] = state_dict.pop('fc.bias')
            
            logger.info(f"键名数量: {len(state_dict)}")
            if state_dict:
                sample_keys = list(state_dict.keys())[:5]
                logger.info(f"样本键名: {sample_keys}")
        
        # 从 checkpoint 获取配置
        config = None
        checkpoint_config = None
        if isinstance(checkpoint, dict):
            # 尝试多种可能的 config 键名
            for config_key in ['config', 'model_config', 'hparams', 'hyper_parameters']:
                if config_key in checkpoint:
                    checkpoint_config = checkpoint[config_key]
                    config = AutoConfig.from_dict(checkpoint_config)
                    logger.info(f"找到 config，键名: {config_key}, vocab_size={config.vocab_size}")
                    break
        
        if config is None:
            # Checkpoint 中没有 config，尝试从 repo 加载
            try:
                logger.info("从 HF repo 加载 config...")
                config = AutoConfig.from_pretrained(REPO_ID)
                logger.info(f"从 repo 加载 config 成功: vocab_size={config.vocab_size}")
            except Exception as e:
                logger.warning(f"无法从 repo 加载 config: {e}")
                config = AutoConfig.from_pretrained("bert-base-chinese")
        
        # 从 state_dict 获取正确的 vocab_size（防止 checkpoint 训练时用了不同 tokenizer）
        checkpoint_vocab_size = None
        for key in state_dict.keys():
            if 'word_embeddings.weight' in key:
                checkpoint_vocab_size = state_dict[key].shape[0]
                break
        
        if checkpoint_vocab_size and config.vocab_size != checkpoint_vocab_size:
            logger.info(f"调整 vocab_size: {config.vocab_size} -> {checkpoint_vocab_size}")
            config.vocab_size = checkpoint_vocab_size
        
        config.num_labels = 11
        logger.info(f"最终模型配置: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, num_labels={config.num_labels}")
        
        # 创建模型结构
        model_status.model_classifier = AutoModelForSequenceClassification.from_config(config)
        
        # 加载权重并检查匹配情况（使用 strict=False 并捕获详细错误）
        try:
            load_result = model_status.model_classifier.load_state_dict(state_dict, strict=False)
            missing_keys = load_result.missing_keys if hasattr(load_result, 'missing_keys') else []
            unexpected_keys = load_result.unexpected_keys if hasattr(load_result, 'unexpected_keys') else []
            
            if missing_keys:
                logger.warning(f"缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"多余的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        except Exception as load_err:
            # 捕获详细的尺寸不匹配错误
            logger.error(f"权重加载详细错误: {load_err}")
            # 打印 embedding 层的实际尺寸
            for key in ['bert.embeddings.word_embeddings.weight', 'embeddings.word_embeddings.weight']:
                if key in state_dict:
                    logger.error(f"Checkpoint {key} shape: {state_dict[key].shape}")
            raise
        
        model_status.model_classifier.eval()
        logger.info(f"分类模型加载成功! (num_labels={config.num_labels})")
        
        # 加载 tokenizer（需要与模型训练时使用相同的 tokenizer）
        tokenizer_loaded = False
        try:
            model_status.tokenizer_classifier = AutoTokenizer.from_pretrained(REPO_ID)
            tokenizer_loaded = True
            logger.info(f"Tokenizer 从 repo 加载成功: {type(model_status.tokenizer_classifier).__name__}")
        except Exception as e:
            logger.warning(f"无法从 repo 加载 tokenizer: {e}")
        
        if not tokenizer_loaded:
            # 使用 hfl/chinese-roberta-wwm-ext（vocab_size=21128，与 checkpoint 匹配）
            model_status.tokenizer_classifier = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            logger.info(f"使用 fallback tokenizer: hfl/chinese-roberta-wwm-ext")
        
        tokenizer_vocab_size = len(model_status.tokenizer_classifier)
        logger.info(f"Tokenizer vocab_size: {tokenizer_vocab_size}")
        
        model_status.classifier_loaded = True
        logger.info("分类模型加载成功!")
        
    except Exception as e:
        logger.error(f"分类模型加载失败: {e}")
        # Fallback: 直接从 HF 加载完整模型
        try:
            model_status.model_classifier = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
            model_status.tokenizer_classifier = AutoTokenizer.from_pretrained(REPO_ID)
            model_status.model_classifier.eval()
            model_status.classifier_loaded = True
            logger.info("分类模型(Fallback)加载成功")
        except Exception as e2:
            logger.error(f"分类模型 Fallback 也失败: {e2}")
    
    # ==========================================
    # 生成模型加载 - mT5 small + 浙大微调 checkpoint
    # ==========================================
    logger.info("-" * 30)
    logger.info("步骤: 加载 mT5-Small 生成模型 + checkpoint...")
    try:
        # 1. 下载浙大微调 checkpoint
        from huggingface_hub import hf_hub_download
        
        gen_ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="rewrite_mT5_small.ckpt",
            token=HF_TOKEN or None
        )
        logger.info(f"Checkpoint 已下载: {gen_ckpt_path}")
        
        # 加载 checkpoint
        raw_ckpt = torch.load(gen_ckpt_path, map_location="cpu", weights_only=False)
        
        # 提取 state_dict
        if isinstance(raw_ckpt, dict):
            if "state_dict" in raw_ckpt:
                state_dict = raw_ckpt["state_dict"]
            elif "model_state_dict" in raw_ckpt:
                state_dict = raw_ckpt["model_state_dict"]
            else:
                state_dict = raw_ckpt
        else:
            state_dict = raw_ckpt
        
        logger.info(f"Checkpoint 原始键名样本: {list(state_dict.keys())[:5]}")
        
        # 清理键名前缀
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "")
            cleaned_state_dict[name] = v
        
        # 2. 加载 mT5 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        logger.info(f"Tokenizer vocab_size: {len(tokenizer)}")
        
        # 测试 tokenizer
        test_text = "共享"
        test_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(test_ids)
        logger.info(f"Tokenizer 测试: '{test_text}' -> {test_ids} -> '{decoded}'")
        
        # 3. 加载 mT5 模型
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        logger.info(f"mT5-base vocab_size: {model.config.vocab_size}")
        
        # 4. 加载微调权重
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        logger.info(f"权重加载: 缺失={len(result.missing_keys)}, 多余={len(result.unexpected_keys)}")
        
        if len(result.missing_keys) > 10:
            logger.error(f"权重加载失败！缺失 {len(result.missing_keys)} 个键")
        
        model.eval()
        model_status.model_generator = model
        model_status.tokenizer_generator = tokenizer
        model_status.generator_loaded = True
        logger.info("mT5 生成模型加载成功!")
        
    except Exception as e:
        logger.error(f"mT5 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 50)
    logger.info("模型加载完成!")
    logger.info(f"  - 分类模型: {'已加载' if model_status.classifier_loaded else '未加载'}")
    logger.info(f"  - 生成模型: {'已加载' if model_status.generator_loaded else '未加载'}")
    logger.info("=" * 50)

# ==========================================
# RAG 组件初始化
# ==========================================
legal_kb_loader: Optional[LegalKBLoader] = None
vector_store: Optional[VectorStore] = None
retriever: Optional[Retriever] = None

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
    except Exception as e:
        logger.error(f"RAG 初始化失败: {e}")
        # 不阻断主程序继续运行

# ==========================================
# FastAPI 应用设置
# ==========================================
app = FastAPI(
    title="隐私政策合规审查 API",
    description="基于 BERT-MoE 和 RAG 的隐私政策合规审查系统",
    version="1.0.0"
)

# CORS 配置 - 允许前端域名访问
ALLOWED_ORIGINS = [
    "https://sy-s-web-3.pages.dev",  # Cloudflare Pages 主域名
    "http://localhost:5000",
    "http://localhost:5173",
]

# Cloudflare Pages 预览域名检查函数
def is_allowed_origin(origin: str) -> bool:
    if origin in ALLOWED_ORIGINS:
        return True
    # 允许 Cloudflare Pages 预览域名 (*.sy-s-web-3.pages.dev)
    if origin.endswith(".sy-s-web-3.pages.dev"):
        return True
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://[a-z0-9-]+\.sy-s-web-3\.pages\.dev|https://sy-s-web-3\.pages\.dev|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
init_db()

# 注册认证路由
app.include_router(auth_router)

# 注册诊断路由
try:
    from diagnose_api import router as diagnose_router
    app.include_router(diagnose_router)
    logger.info("诊断 API 已注册")
except ImportError as e:
    logger.warning(f"诊断 API 加载失败: {e}")

# ==========================================
# 合规指标体系与权重定义
# ==========================================
INDICATORS = {
    "过度收集敏感数据": {"weight": 0.15, "id": "I1", 
                        "legal_basis": "《个人信息保护法》第28、29条;《网络安全法》第41条"},
    "未说明收集目的": {"weight": 0.12, "id": "I2",
                      "legal_basis": "《个人信息保护法》第17条;《个人信息安全规范》第5.1条"},
    "未获得明示同意": {"weight": 0.15, "id": "I3",
                      "legal_basis": "《个人信息保护法》第14、15条;《网络安全法》第41条"},
    "收集范围超出服务需求": {"weight": 0.10, "id": "I4",
                           "legal_basis": "《个人信息保护法》第6条(最小必要原则)"},
    "未明确第三方共享范围": {"weight": 0.08, "id": "I5",
                           "legal_basis": "《个人信息保护法》第23条;《个人信息安全规范》第8.1条"},
    "未获得单独共享授权": {"weight": 0.12, "id": "I6",
                         "legal_basis": "《个人信息保护法》第23、29条"},
    "未明确共享数据用途": {"weight": 0.08, "id": "I7",
                         "legal_basis": "《个人信息保护法》第6、23条;GDPR第46条"},
    "未明确留存期限": {"weight": 0.05, "id": "I8",
                      "legal_basis": "《个人信息保护法》第19条;《个人信息安全规范》第6.1条"},
    "未说明数据销毁机制": {"weight": 0.05, "id": "I9",
                         "legal_basis": "《个人信息保护法》第19、47条"},
    "未明确用户权利范围": {"weight": 0.05, "id": "I10",
                         "legal_basis": "《个人信息保护法》第44-48条"},
    "未提供便捷权利行使途径": {"weight": 0.03, "id": "I11",
                             "legal_basis": "《个人信息保护法》第50条;《个人信息安全规范》第7.9条"},
    "未明确权利响应时限": {"weight": 0.02, "id": "I12",
                         "legal_basis": "《个人信息安全规范》第8.10条"}
}

# 创建 ID 到指标名称的映射
ID_TO_INDICATOR = {v["id"]: k for k, v in INDICATORS.items()}
INDICATOR_KEYS = list(INDICATORS.keys())

# ==========================================
# Pydantic 请求/响应模型 (内联定义，避免重复)
# ==========================================
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000, description="隐私政策文本")
    source_type: str = Field(default="text", description="来源类型")

class AnalyzeResponse(BaseModel):
    id: str
    name: str
    score: float
    risk_level: str
    violations: List[Dict[str, Any]]
    created_at: Optional[str] = None

class RectifyRequest(BaseModel):
    original_snippet: str = Field(..., min_length=5, description="违规条款原文", alias="original_snippet")
    violation_type: str = Field(..., description="违规类型ID，如 I1")

class UrlRequest(BaseModel):
    url: str = Field(..., description="目标URL")

# ==========================================
# 辅助函数
# ==========================================
def split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子（只按句号/换行，不按逗号/分号）"""
    # 只按句号、感叹号、问号、换行切分，保留完整句子
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]  # 至少10字符才保留

def roberta_predict(sentence: str) -> tuple:
    """使用 RoBERTa 分类模型预测，返回 (probs, confidence)"""
    if not model_status.classifier_loaded:
        logger.warning("分类模型未加载，返回默认概率")
        return [0.0] * 11, 0.0
    
    inputs = model_status.tokenizer_classifier(
        sentence, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model_status.model_classifier(**inputs)
        logits = outputs.logits.squeeze()
        logits_list = logits.tolist()
        
        # 调试日志：输出原始 logits
        logger.info(f"原始 logits: {logits_list}")
        
        # 【关键修复】使用 logits 差值代替 softmax
        # softmax 是相对概率，当 logits 差距不够大时概率偏低
        # 使用 max(logits) - mean(logits) 作为置信度，更直观
        logits_tensor = logits if hasattr(logits, '__iter__') else logits.unsqueeze(0)
        max_logit = torch.max(logits_tensor)
        mean_logit = torch.mean(logits_tensor)
        confidence = (max_logit - mean_logit).item()
        
        # 同时计算 softmax 概率（用于多标签检测）
        probs = torch.softmax(logits_tensor, dim=-1).tolist()
        if not isinstance(probs, list):
            probs = [probs]
    
    # 调试日志：输出概率分布
    max_idx = probs.index(max(probs)) if probs else -1
    logger.info(f"句子: {sentence[:30]}... | 最高类别: {max_idx}, 置信度: {confidence:.4f}, softmax概率: {probs[max_idx] if max_idx >= 0 else 0:.4f}")
    
    return probs, confidence

# 违规类型对应的法律关键词（用于引导模型改写）
LEGAL_KEYWORDS = {
    "I1": "最小必要原则",
    "I2": "明确告知目的",
    "I3": "取得用户同意",
    "I4": "敏感信息单独同意",
    "I5": "第三方共享告知",
    "I6": "单独授权",
    "I7": "明确存储期限",
    "I8": "合理范围收集",
    "I9": "及时删除机制",
    "I10": "用户权利保障",
    "I11": "便捷投诉渠道",
    "I12": "及时响应",
}


def get_legal_keywords(violation_type: str, context: Optional[str] = None) -> str:
    """从 RAG 检索结果中提取法律 action_tag（核心合规动作）"""
    # 如果 RAG 不可用，使用静态关键词
    if not RAG_AVAILABLE or retriever is None:
        return LEGAL_KEYWORDS.get(violation_type, "合法合规")
    
    try:
        results = retriever.retrieve_by_violation_type(violation_type, context=context, top_k=1)
        if results:
            result = results[0]
            # 优先取 keywords_matched[0] 作为 action_tag（最精炼的合规动作词）
            if result.keywords_matched:
                return result.keywords_matched[0]
    except Exception:
        pass
    
    # 回退到静态关键词
    return LEGAL_KEYWORDS.get(violation_type, "合法合规")


def get_legal_basis_from_rag(violation_type: str, context: Optional[str] = None) -> str:
    """使用 RAG 检索获取法律依据（包含完整条款内容）"""
    if not RAG_AVAILABLE or retriever is None:
        # 回退到静态配置
        for name, info in INDICATORS.items():
            if info["id"] == violation_type:
                return info["legal_basis"]
        return "《个人信息保护法》"
    
    try:
        results = retriever.retrieve_by_violation_type(violation_type, context=context, top_k=2)
        logger.info(f"RAG检索 {violation_type}: 获得 {len(results)} 条结果")
        
        if results:
            legal_refs = []
            for result in results[:2]:
                # 返回完整法律引用和条款内容（增加展示长度）
                full_ref = f"{result.law} {result.article_number}：{result.content[:200]}"
                logger.info(f"  -> {full_ref[:100]}...")
                legal_refs.append(full_ref)
            return "；".join(legal_refs) if legal_refs else INDICATORS.get(
                ID_TO_INDICATOR.get(violation_type, ""), {}
            ).get("legal_basis", "《个人信息保护法》")
        else:
            logger.warning(f"RAG检索 {violation_type} 返回空，使用fallback")
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
    
    return INDICATORS.get(ID_TO_INDICATOR.get(violation_type, ""), {}).get(
        "legal_basis", "《个人信息保护法》"
    )

def calculate_compliance_score(violation_ids: List[str]) -> tuple[float, str]:
    """
    计算合规评分
    
    公式: S = 100 - Σ(wi × vi × 100)
    
    Args:
        violation_ids: 违规类型ID列表
        
    Returns:
        (总分, 风险等级)
    """
    penalty = 0.0
    for ind_name, info in INDICATORS.items():
        if info["id"] in violation_ids:
            penalty += info["weight"] * 1.0  # vi=1 表示违规
    
    total_score = round(max(0.0, 100.0 - (penalty * 100.0)), 1)
    
    if total_score >= 70:
        risk_level = "低风险"
    elif 40 <= total_score < 70:
        risk_level = "中等风险"
    else:
        risk_level = "高风险"
    
    return total_score, risk_level

# ==========================================
# API 端点
# ==========================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_classifier": "loaded" if model_status.classifier_loaded else "not_loaded",
        "model_generator": "loaded" if model_status.generator_loaded else "not_loaded",
        "rag_available": RAG_AVAILABLE,
        "rag_initialized": retriever is not None if RAG_AVAILABLE else False,
        "port": PORT
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
    """
    分析隐私政策文本
    
    检测流程:
    1. 文本分句
    2. BERT-MoE 分类 (11类)
    3. 11类→12类映射
    4. RAG 法律依据检索
    5. 计算合规评分
    """
    sentences = split_into_sentences(request.text)
    violations_list = []
    all_violation_ids = {}  # 改为dict，保留每个违规类型的所有句子
    all_snippets = []  # 所有句子及其分类结果
    
    for idx, sentence in enumerate(sentences):
        # BERT-MoE 分类预测
        probs, confidence = roberta_predict(sentence)
        
        # 11类 → 12类 映射
        if map_to_12_classes is not None:
            violation_ids = map_to_12_classes(probs, confidence=confidence)
        else:
            ID_MAPPING_FALLBACK = {
                0: "I1", 1: "I2", 2: "I3", 3: "I4", 4: "I5",
                5: "I6", 6: "I7", 7: "I8", 8: "I9", 9: "I10", 10: "I11"
            }
            violation_ids = []
            if probs:
                max_idx = probs.index(max(probs))
                max_prob = probs[max_idx]
                if confidence is not None and confidence >= 1.8:
                    v_id = ID_MAPPING_FALLBACK.get(max_idx)
                    v_id = ID_MAPPING_FALLBACK.get(max_idx)
                    if v_id:
                        violation_ids = [v_id]
        
        # 记录所有句子的分类结果
        max_idx = probs.index(max(probs)) if probs else 0
        max_prob = probs[max_idx] if probs else 0
        all_snippets.append({
            "id": idx,
            "text": sentence,
            "predicted_class": max_idx,
            "predicted_class_name": ["数据收集", "权限获取", "共享转让", "使用", "存储方式", "安全措施", "特殊人群", "权限管理", "联系方式", "政策变更", "停止运营"][max_idx] if max_idx < 11 else "其他",
            "confidence": round(max_prob, 4),
            "violation_ids": violation_ids
        })
        
        for v_id in violation_ids:
            indicator_name = ID_TO_INDICATOR.get(v_id)
            if indicator_name:
                # 使用dict保留每个违规类型的所有句子（不去重）
                if v_id not in all_violation_ids:
                    all_violation_ids[v_id] = []
                
                # RAG 获取法律依据
                legal_basis = get_legal_basis_from_rag(v_id, context=sentence)
                
                all_violation_ids[v_id].append({
                    "indicator": indicator_name,
                    "violation_name": indicator_name,
                    "violation_id": v_id,
                    "category_name": indicator_name,
                    "snippet": sentence,
                    "legal_basis": legal_basis,
                    "confidence": round(max_prob, 4)
                })
    
    # 汇总所有违规（保留每个违规类型下的所有句子）
    for v_id, v_list in all_violation_ids.items():
        violations_list.extend(v_list)
    
    # 计算合规评分（基于违规类型数量）
    score, risk_level = calculate_compliance_score(list(all_violation_ids.keys()))
    
    # 生成项目
    project_id = f"p{int(datetime.utcnow().timestamp())}"
    project = Project(
        id=project_id,
        user_id=current_user.id,
        name=f"审查-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
        source_type=request.source_type,
        score=score,
        risk_level=risk_level,
        result_json=json.dumps(violations_list),
        raw_text=request.text[:5000] if len(request.text) > 5000 else request.text
    )
    db.add(project)
    db.commit()
    
    return {
        "id": project.id,
        "name": project.name,
        "score": score,
        "risk_level": risk_level,
        "violations": violations_list,
        "all_snippets": all_snippets,  # 返回所有句子的分类结果
        "created_at": project.created_at.isoformat()
    }

@app.post("/api/v1/rectify")
async def rectify_snippet(
    request: RectifyRequest,
    current_user: User = Depends(get_current_user)
):
    """生成违规条款的整改建议"""
    # 获取指标名称
    indicator_name = ID_TO_INDICATOR.get(request.violation_type, "")
    
    # 使用 mT5 生成整改建议
    if model_status.generator_loaded:
        logger.info(f"========== 整改生成开始 ==========")
        
        # RAG 检索获取法条，手动映射为 action_tag
        legal_context = get_legal_basis_from_rag(request.violation_type, context=f"{indicator_name} {request.original_snippet}")
        
        # 手动映射 RAG 法条 → action_tag（翻译成模型能听懂的指令）
        ACTION_TAGS = {
            "I1": "最小必要收集",           # 过度收集 → 最小必要
            "I2": "明确告知目的",            # 未说明目的 → 告知目的
            "I3": "取得同意授权",            # 未获同意 → 取得同意
            "I4": "单独明示同意",            # 敏感信息 → 单独同意
            "I5": "告知共享目的并取得同意",   # 第三方共享 → 告知并同意
            "I6": "单独授权处理",            # 敏感授权 → 单独授权
            "I7": "明确保存期限",            # 未定期限 → 明确期限
            "I8": "数据最小化",             # 超范围收集 → 数据最小化
            "I9": "增加安全措施说明",        # 未说明安全 → 说明措施
            "I10": "明确用户权利",           # 未保障权利 → 明确权利
            "I11": "提供便捷投诉",           # 无投诉渠道 → 提供渠道
            "I12": "规定响应时限",           # 未响应时限 → 及时响应
        }
        action_tag = ACTION_TAGS.get(request.violation_type, "合规改写")
        
        # Prompt：[微调指令] [RAG提取出的短动作] [原文] [人工起始符]
        prompt = f"summarization: [合规改写方向：{action_tag}] 原文：{request.original_snippet[:50]}。重写为通俗合规的表述："
        
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = model_status.tokenizer_generator(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        
        # 生成 - 暴力惩罚复读，强制寻找改写路径
        with torch.no_grad():
            logger.info("开始调用 model.generate()...")
            output_ids = model_status.model_generator.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,           
                num_beams=10,               # 增加搜索深度
                no_repeat_ngram_size=3,      # 限制连续重复
                repetition_penalty=2.2,       # 暴力惩罚复读
                length_penalty=0.8,           # 防止瞎编长句
                early_stopping=True,
                num_return_sequences=1,
            )
            logger.info(f"生成完成! output_ids shape: {output_ids.shape}")
        
        # 解码
        tokenizer = model_status.tokenizer_generator
        logger.info(f"Tokenizer vocab_size: {len(tokenizer)}")
        
        raw_result = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 去掉所有空格
        suggested_text = ''.join(raw_result.split())
        
        logger.info(f"原始解码结果: '{raw_result}'")
        logger.info(f"后处理后结果: '{suggested_text}'")
        logger.info(f"========== 整改生成结束 ==========")
    else:
        # Fallback: 返回通用建议
        suggested_text = f"建议修改条款内容，确保符合{indicator_name}的合规要求。"
    
    # 计算diff
    dmp = diff_match_patch()
    diffs = dmp.diff_main(request.original_snippet, suggested_text)
    dmp.diff_cleanupSemantic(diffs)
    
    # 生成HTML
    diff_original_parts = []
    diff_suggested_parts = []
    for op, text in diffs:
        if op == -1:  # 删除
            diff_original_parts.append(f'<span class="diff-remove">{text}</span>')
        elif op == 0:  # 相同
            diff_original_parts.append(text)
            diff_suggested_parts.append(text)
        else:  # 添加
            diff_suggested_parts.append(f'<span class="diff-add">{text}</span>')
    
    diff_original_html = ''.join(diff_original_parts)
    diff_suggested_html = ''.join(diff_suggested_parts)
    
    return {
        "suggested_text": suggested_text,
        "legal_basis": legal_context,
        "diff_original_html": diff_original_html,
        "diff_suggested_html": diff_suggested_html
    }

@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """上传文件并提取文本"""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    return {"text": text}

@app.post("/api/v1/fetch-url")
async def fetch_url(
    request: UrlRequest,
    current_user: User = Depends(get_current_user)
):
    """抓取URL内容"""
    import requests
    from bs4 import BeautifulSoup
    
    # 验证URL格式
    if not request.url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400, 
            detail="请输入以 http:// 或 https:// 开头的有效URL"
        )
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(request.url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # 检查内容类型
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            raise HTTPException(
                status_code=400,
                detail=f"该URL返回的内容不是网页格式 ({content_type})，请确认输入的是隐私政策网页URL"
            )
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除 script 和 style 标签
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取文本
        text = soup.get_text(separator='\n', strip=True)
        
        # 清理空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 检查是否抓取到有效内容
        if len(text) < 100:
            raise HTTPException(
                status_code=400,
                detail="抓取到的内容过少，可能是页面需要登录或存在验证码，请尝试直接复制网页文本内容进行分析"
            )
        
        return {"text": text}
        
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=400,
            detail="请求超时，请检查URL是否可访问，或尝试更换网络后重试"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=400,
            detail="无法连接到该URL，请确认URL是否正确（如：https://xxx.com/privacy）"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"无法读取该网页内容，请确认URL是否正确，或尝试直接复制网页文本进行分析"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="网页抓取失败，请尝试直接复制网页文本内容进行分析"
        )

@app.get("/api/v1/projects")
async def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的项目列表"""
    projects = db.query(Project).filter(
        Project.user_id == current_user.id
    ).order_by(Project.created_at.desc()).limit(50).all()
    
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
    """获取项目详情"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
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

@app.get("/api/v1/export/{project_id}")
async def export_report(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """导出项目报告"""
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    violations = json.loads(project.result_json) if project.result_json else []
    
    report = f"""隐私政策合规审查报告
{'=' * 50}

项目名称：{project.name}
审查时间：{project.created_at.strftime('%Y-%m-%d %H:%M')}
合规得分：{project.score}
风险等级：{project.risk_level}

{'=' * 50}
违规条款统计
{'-' * 50}
共发现 {len(violations)} 项潜在风险

{'=' * 50}
详细分析
{'-' * 50}
"""
    for i, v in enumerate(violations, 1):
        report += f"\n{i}. {v.get('indicator', '未知类别')} (ID: {v.get('violation_id', 'N/A')})\n"
        report += f"   原文：{v.get('snippet', '未知')}\n"
        report += f"   依据：{v.get('legal_basis', '未知')}\n"
    
    report += f"\n{'=' * 50}\n"
    report += "报告生成时间：{}\n".format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
    report += "本报告由隐私政策合规审查系统自动生成\n"
    
    return Response(
        content=report,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''report_{project_id}.txt"}
    )

# ==========================================
# 启动事件
# ==========================================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info(f"API 服务启动中... (端口: {PORT})")
    
    # 加载模型
    load_models()
    
    # 初始化 RAG
    initialize_rag()
    
    logger.info("API 服务启动完成!")

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
