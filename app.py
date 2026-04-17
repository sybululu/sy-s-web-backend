"""
隐私政策合规审查 API
整合了 RAG 架构的法律知识库检索
"""
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MT5ForConditionalGeneration

from models import User, Project, get_db, init_db, Article, RetrievedChunk
from auth import router as auth_router, get_current_user

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 导入 RAG 模块
# ==========================================
try:
    from src.loader import LegalKBLoader
    from src.store import VectorStore
    from src.search import Retriever
    from src.config import get_config
    RAG_AVAILABLE = True
    logger.info("RAG 模块加载成功")
except ImportError as e:
    logger.warning(f"RAG 模块加载失败: {e}")
    RAG_AVAILABLE = False

# ==========================================
# 模型加载 (HuggingFace Transformers)
# ==========================================
print("正在加载模型...")

# HuggingFace Hub 配置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "sybululu/bert-moe")

# HuggingFace Inference API 配置 (Phi-4 Mini)
# 设置 USE_HF_API=1 启用 API 调用模式
USE_HF_API = os.environ.get("USE_HF_API", "0") == "1"
HF_INFERENCE_MODEL = os.environ.get("HF_INFERENCE_MODEL", "microsoft/phi-4-mini-instruct")

# 1. 加载 RoBERTa 风险分类模型 (sybululu/bert-moe)
# 你的模型缺少 config.json，需要手动构建架构
from transformers import BertConfig, BertForSequenceClassification
from huggingface_hub import hf_hub_download
import torch

tokenizer_roberta = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 从基础模型获取配置
base_config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext")
base_config.num_labels = 12

# 手动创建模型架构
model_roberta = BertForSequenceClassification(config=base_config)

# 尝试加载权重文件
try:
    # 先尝试 safetensors 格式
    model_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="model.safetensors",
        token=HF_TOKEN or None
    )
    from safetensors.torch import load_file
    state_dict = load_file(model_file)
    model_roberta.load_state_dict(state_dict, strict=False)
    print(f"✓ 分类模型加载成功: {HF_REPO_ID}")
except Exception as e:
    try:
        # 降级到 PyTorch 格式
        model_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="pytorch_model.bin",
            token=HF_TOKEN or None
        )
        state_dict = torch.load(model_file, map_location="cpu")
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model_roberta.load_state_dict(state_dict, strict=False)
        print(f"✓ 分类模型加载成功: {HF_REPO_ID}")
    except Exception as e2:
        raise RuntimeError(f"无法从 {HF_REPO_ID} 加载模型权重: {e2}")

model_roberta.eval()

# mT5 降级模型 - 延迟加载，只有在 API 失败时才加载
_model_mt5_cache = {"model": None, "tokenizer": None}

def get_fallback_model():
    """延迟加载 mT5 降级模型，避免占用内存"""
    global model_mt5, tokenizer_mt5
    
    if _model_mt5_cache["model"] is None:
        try:
            _model_mt5_cache["tokenizer"] = AutoTokenizer.from_pretrained("google/mt5-base")
            _model_mt5_cache["model"] = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
            _model_mt5_cache["model"].eval()
            print("✓ mT5 降级模型加载成功（延迟加载）")
        except Exception as e:
            logger.warning(f"mT5 模型加载失败: {e}")
            return None, None
    
    return _model_mt5_cache["model"], _model_mt5_cache["tokenizer"]

# 初始化为 None，启动时不加载
model_mt5 = None
tokenizer_mt5 = None

print("模型加载完成！")

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
app = FastAPI(title="隐私政策合规审查 API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sy-s-web.pages.dev", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
init_db()

# 注册认证路由
app.include_router(auth_router)

# ==========================================
# 合规指标体系与权重定义
# ==========================================
INDICATORS = {
    "过度收集敏感数据": {"weight": 0.15, "legal_basis": "《个人信息保护法》第六条'最小必要'原则及第二十九条", "id": "I1"},
    "未说明收集目的": {"weight": 0.12, "legal_basis": "《个人信息保护法》第十七条", "id": "I2"},
    "未获得明示同意": {"weight": 0.15, "legal_basis": "《个人信息保护法》第十四条", "id": "I3"},
    "收集范围超出服务需求": {"weight": 0.10, "legal_basis": "《个人信息保护法》第六条", "id": "I4"},
    "未明确第三方共享范围": {"weight": 0.08, "legal_basis": "《个人信息保护法》第二十三条", "id": "I5"},
    "未获得单独共享授权": {"weight": 0.12, "legal_basis": "《个人信息保护法》第二十三条", "id": "I6"},
    "未明确共享数据用途": {"weight": 0.08, "legal_basis": "《个人信息保护法》第二十三条及GDPR第四十六条", "id": "I7"},
    "未明确留存期限": {"weight": 0.05, "legal_basis": "《个人信息保护法》第十九条", "id": "I8"},
    "未说明数据销毁机制": {"weight": 0.05, "legal_basis": "《个人信息保护法》第四十七条", "id": "I9"},
    "未明确用户权利范围": {"weight": 0.05, "legal_basis": "《个人信息保护法》第四十四至四十八条", "id": "I10"},
    "未提供便捷权利行使途径": {"weight": 0.03, "legal_basis": "《个人信息保护法》第五十条", "id": "I11"},
    "未明确权利响应时限": {"weight": 0.02, "legal_basis": "《个人信息安全规范》GB/T 35273-2020", "id": "I12"}
}

# 创建 ID 到指标名称的映射
ID_TO_INDICATOR = {v["id"]: k for k, v in INDICATORS.items()}
INDICATOR_KEYS = list(INDICATORS.keys())

# ==========================================
# Pydantic 数据模型定义 (Schema)
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    source_type: Optional[str] = "text"
    
    @validator('text')
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

class RectifyRequest(BaseModel):
    original_snippet: str
    violation_type: str
    legal_basis: Optional[str] = None  # 可选的法律依据，用于前端传递

class UrlRequest(BaseModel):
    url: str

# ==========================================
# 辅助函数
# ==========================================
def split_into_sentences(text: str) -> List[str]:
    # 使用正向预查保留标点，避免句子末尾标点丢失
    sentences = re.split(r'(?<=[。；\n])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def roberta_predict(sentence: str) -> Dict[str, float]:
    """预测句子是否包含违规"""
    if model_roberta is None or tokenizer_roberta is None:
        # 模型未加载时返回空结果
        return {key: 0.0 for key in INDICATOR_KEYS}
    
    inputs = tokenizer_roberta(sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_roberta(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    if not isinstance(probs, list):
        probs = [probs]
        
    return {INDICATOR_KEYS[i]: probs[i] for i in range(min(len(probs), len(INDICATOR_KEYS)))}

def get_legal_basis_from_rag(violation_type: str, context: Optional[str] = None) -> str:
    """
    使用 RAG 检索获取法律依据
    
    Args:
        violation_type: 违规类型ID，如 "I1"
        context: 违规上下文描述
        
    Returns:
        检索到的法律依据文本
    """
    if not RAG_AVAILABLE or retriever is None:
        # 回退到静态配置
        for name, info in INDICATORS.items():
            if info["id"] == violation_type:
                return info["legal_basis"]
        return "《个人信息保护法》"
    
    try:
        results = retriever.retrieve_by_violation_type(violation_type, context=context, top_k=3)
        if results:
            # 合并检索结果
            legal_refs = []
            for result in results[:2]:
                ref = f"{result.law}{result.article_number}"
                legal_refs.append(ref)
            return "；".join(legal_refs) if legal_refs else INDICATORS.get(ID_TO_INDICATOR.get(violation_type, ""), {}).get("legal_basis", "《个人信息保护法》")
    except Exception as e:
        logger.error(f"RAG 检索失败: {e}")
    
    return INDICATORS.get(ID_TO_INDICATOR.get(violation_type, ""), {}).get("legal_basis", "《个人信息保护法》")

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
    violation_flags = {key: 0 for key in INDICATOR_KEYS}
    violations_list = []

    for sentence in sentences:
        probs = roberta_predict(sentence)
        for indicator, prob in probs.items():
            if prob > 0.5:
                violation_flags[indicator] = 1
                # 记录所有违规句子，不再限制只报一次
                # 这样用户能看到所有违规位置，便于逐一整改
                violation_id = INDICATORS[indicator]["id"]
                legal_basis = get_legal_basis_from_rag(violation_id, context=sentence)
                
                violations_list.append({
                    "indicator": indicator,
                    "violation_id": violation_id,
                    "snippet": sentence,
                    "legal_basis": legal_basis
                })

    penalty = sum(INDICATORS[ind]["weight"] * vi for ind, vi in violation_flags.items())
    total_score = round(max(0.0, 100.0 - (penalty * 100.0)), 1)

    if total_score >= 70:
        risk_level = "低风险"
    elif 40 <= total_score < 70:
        risk_level = "中等风险"
    else:
        risk_level = "高风险"

    project_id = f"p{int(datetime.utcnow().timestamp())}"
    project = Project(
        id=project_id,
        user_id=current_user.id,
        name=f"审查-{datetime.utcnow().strftime('%Y%m%d')}",
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
        "violations": violations_list
    }

@app.post("/api/v1/rectify")
async def rectify_snippet(
    request: RectifyRequest,
    current_user: User = Depends(get_current_user)
):
    """整改违规条款 - 使用 HuggingFace Inference API (Phi-4 Mini) 生成合规改写"""
    
    # 违规类型提示
    violation_type_hints = {
        "I1": "涉及收集个人信息，必须遵守最小必要原则，只能收集与服务直接相关的个人信息，禁止收集与服务无关的敏感信息。",
        "I2": "必须明确说明每项个人信息收集的具体目的和用途，不能使用模糊表述。",
        "I3": "涉及处理个人信息必须获得用户明确、知情、自愿的同意，不能捆绑授权。",
        "I4": "收集范围不得超过实现处理目的的最小必要范围。",
        "I5": "向第三方共享时必须明确说明接收方类型、共享目的、数据类型，禁止无限制共享。",
        "I6": "向第三方提供个人信息必须单独取得用户明示同意。",
        "I7": "必须明确说明第三方使用数据的目的和范围。",
        "I8": "必须明确数据存储期限，期限届满应予以删除或匿名化。",
        "I9": "必须说明数据销毁机制，承诺在约定保存期限届满后主动删除或匿名化处理。",
        "I10": "必须明确列举用户享有的各项权利及行使方式。",
        "I11": "必须提供便捷的渠道供用户行使权利，渠道必须易于发现和操作。",
        "I12": "必须明确权利响应时限，承诺在法定期限内处理用户请求。",
    }
    
    violation_hint = violation_type_hints.get(request.violation_type, "必须符合《个人信息保护法》相关要求。")
    
    # 获取 RAG 法律依据
    legal_context = request.legal_basis if request.legal_basis else get_legal_basis_from_rag(request.violation_type, context=request.original_snippet)
    
    # 提取法律关键词用于增强 prompt，传入 violation_id 做兜底
    legal_keywords = extract_legal_keywords(legal_context, request.violation_type)
    
    suggested_text = ""
    
    # 优先使用 HuggingFace Inference API
    if USE_HF_API and HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(model=HF_INFERENCE_MODEL, token=HF_TOKEN)
            
            # Phi-4 Mini Instruct 格式
            messages = [
                {"role": "system", "content": "你是一位资深的隐私合规专家。你需要根据提供的[法律依据]和[整改要求]，将[原句]重写为符合法律规范的表述。重写时必须：\n1. 遵循最小必要原则\n2. 明确说明处理目的\n3. 保障用户知情权和选择权\n4. 语言通俗易懂，避免法律术语堆砌"},
                {"role": "user", "content": f"[法律依据摘要]：{legal_keywords}\n[整改要求]：{violation_hint}\n[原句]：{request.original_snippet}"}
            ]
            
            response = client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            
            suggested_text = response.choices[0].message.content.strip()
            logger.info(f"HF Inference API (Phi-4 Mini) 生成成功")
        except Exception as e:
            logger.error(f"HF Inference API 生成失败: {e}")
            suggested_text = ""
    
    # 降级方案：延迟加载 mT5（仅在 API 失败后加载）
    if not suggested_text:
        fallback_model, fallback_tokenizer = get_fallback_model()
        if fallback_model is not None and fallback_tokenizer is not None:
            try:
                mt5_prompt = f"""请将以下隐私政策条款改写为符合法律规范的版本。
【整改要求】{violation_hint}
【原条款】{request.original_snippet}
【合规改写】"""
                
                inputs = fallback_tokenizer(mt5_prompt, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = fallback_model.generate(**inputs, max_length=256, temperature=0.3, do_sample=True)
                suggested_text = fallback_tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"mT5 降级生成成功")
            except Exception as e:
                logger.error(f"mT5 生成失败: {e}")
                suggested_text = ""
    
    # 最终降级：基于规则的通用建议
    if not suggested_text:
        suggested_text = generate_rule_based_suggestion(request.original_snippet, request.violation_type, violation_hint)
    
    return {
        "suggested_text": suggested_text,
        "legal_basis": legal_context
    }


def extract_legal_keywords(legal_context: str, violation_id: str = None) -> str:
    """从法律依据中提取关键要求
    
    Args:
        legal_context: RAG 检索返回的法律依据文本
        violation_id: 违规类型 ID（如 I1, I2），用于兜底
    """
    if not legal_context:
        # 如果没有检索到法律依据，使用违规类型的语义化描述作为兜底
        if violation_id:
            fallback_map = {
                "I1": "《个人信息保护法》第六条：收集个人信息应当具有明确、合理的目的，并遵循最小必要原则。",
                "I2": "《个人信息保护法》第十七条：处理个人信息应当告知个人处理目的、方式和范围。",
                "I3": "《个人信息保护法》第十三条：处理个人信息应当取得个人的同意。",
                "I4": "《个人信息保护法》第六条：收集个人信息的范围应当与处理目的直接相关。",
                "I5": "《个人信息保护法》第二十三条：向第三方提供个人信息应当告知并取得单独同意。",
                "I6": "《个人信息保护法》第二十三条：向第三方提供个人信息应当取得个人的明示同意。",
                "I7": "《个人信息保护法》第二十三条：应当告知个人第三方使用信息的目的和范围。",
                "I8": "《个人信息保护法》第十九条：个人信息的保存期限应当为实现处理目的所必需的最短时间。",
                "I9": "《个人信息保护法》第十九条：保存期限届满应当予以删除或匿名化处理。",
                "I10": "《个人信息保护法》第四十四条至第四十五条：个人享有查阅、复制、更正、删除等权利。",
                "I11": "《个人信息保护法》第五十条：应当提供便捷的渠道供个人行使权利。",
                "I12": "《个人信息保护法》第五十条：应当在合理期限内处理个人的请求。",
            }
            return fallback_map.get(violation_id, "遵循《个人信息保护法》相关规定")
        return "遵循《个人信息保护法》相关规定"
    
    # 提取法条编号
    import re
    articles = re.findall(r'第[零一二三四五六七八九十百]+[条章节款]', legal_context)
    law_names = re.findall(r'《[^》]+》', legal_context)
    
    # 提取关键动词和要求
    keywords = []
    if "同意" in legal_context:
        keywords.append("获得明确同意")
    if "告知" in legal_context:
        keywords.append("充分告知")
    if "最小" in legal_context or "必要" in legal_context:
        keywords.append("最小必要原则")
    if "目的" in legal_context:
        keywords.append("明确处理目的")
    if "删除" in legal_context or "匿名" in legal_context:
        keywords.append("数据删除/匿名化")
    if "第三方" in legal_context:
        keywords.append("第三方共享限制")
    if "权利" in legal_context:
        keywords.append("用户权利保障")
    if "期限" in legal_context or "时间" in legal_context:
        keywords.append("响应时限")
    
    result = ""
    if law_names:
        result += "、".join(set(law_names)) + "规定："
    if articles:
        result += "、".join(articles[:3]) + "。"
    if keywords:
        result += "核心要求：" + "、".join(keywords[:4])
    
    # 最终兜底：如果什么都没提取到
    if not result and violation_id:
        return extract_legal_keywords("", violation_id)
    
    return result or "遵循个人信息保护相关法律法规"


def generate_rule_based_suggestion(original_text: str, violation_type: str, violation_hint: str) -> str:
    """基于规则的降级建议生成"""
    # 通用改写模板
    templates = {
        "I1": f"为了向您提供[具体服务名称]，我们仅收集实现该服务所必需的个人信息，包括[具体信息类型]。我们不会收集与服务无关的信息。",
        "I2": f"我们收集您的[信息类型]用于[具体明确的目的]，包括[列举用途]。",
        "I3": f"在收集您的个人信息前，我们将明确告知您收集的目的、方式和范围，并获得您的同意。您有权拒绝或撤回同意。",
        "I4": f"我们仅收集实现服务目的所必需的最少个人信息，不收集与服务无关的信息。",
        "I5": f"我们仅在以下情况下与第三方共享您的信息：①取得您的单独同意；②实现服务所必需；③法律法规要求。共享时我们会明确告知接收方类型和目的。",
        "I6": f"向第三方提供您的个人信息前，我们会单独征求您的明示同意。",
        "I7": f"第三方使用您的信息时，必须遵守本隐私政策的约定，仅用于约定的目的和范围。",
        "I8": f"我们将在实现处理目的所必需的最短时间内保存您的个人信息，保存期限届满后将予以删除或匿名化处理。",
        "I9": f"当保存期限届满或您行使删除权时，我们将按照法律法规要求的方式删除或匿名化您的个人信息。",
        "I10": f"您依法享有查阅、复制、更正、删除您的个人信息的权利，以及数据可携带权等。",
        "I11": f"您可以通过[具体渠道，如设置页面、联系邮箱]便捷地行使您的个人信息相关权利。",
        "I12": f"我们将在收到您的权利请求后[法定期限/承诺期限]内进行处理和答复。",
    }
    
    return templates.get(violation_type, f"建议修改为更加明确、具体且符合《个人信息保护法》等相关法规要求的表述。")


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
        suggested = v.get('suggestedText', '')
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
