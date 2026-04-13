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

# CORS 配置 - 支持本地和Cloudflare Pages
CORS_ORIGINS = [
    "https://sy-s-web.pages.dev",
    "http://localhost:3000",
    "http://localhost:5173",
    os.environ.get("FRONTEND_URL", ""),  # 可通过环境变量配置
]

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
            
            # 清理键名前缀
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("model.", "").replace("bert.", "")
                cleaned_state_dict[new_k] = v
            state_dict = cleaned_state_dict
        
        # 从 checkpoint 获取配置，直接创建完整模型
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            config = AutoConfig.from_dict(config_dict)
        else:
            # 使用默认配置
            config = AutoConfig.from_pretrained(REPO_ID)
        config.num_labels = 11
        
        # 创建模型结构
        model_status.model_classifier = AutoModelForSequenceClassification.from_config(config)
        model_status.model_classifier.load_state_dict(state_dict, strict=False)
        model_status.model_classifier.eval()
        
        # 直接从 checkpoint 目录加载 tokenizer（不单独下载 base model）
        try:
            model_status.tokenizer_classifier = AutoTokenizer.from_pretrained(REPO_ID)
        except:
            # Fallback: 使用通用中文 tokenizer
            model_status.tokenizer_classifier = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
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
    
    # 3. 加载生成模型 (mT5 small)
    # checkpoint 包含完整权重，直接加载即可
    logger.info("-" * 30)
    logger.info("步骤2/2: 加载生成模型...")
    try:
        from huggingface_hub import hf_hub_download
        
        # 下载包含完整权重的 checkpoint
        gen_ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="rewrite_mT5_small.ckpt",
            token=HF_TOKEN or None
        )
        logger.info(f"生成模型 checkpoint 已下载: {gen_ckpt_path}")
        
        # 加载 checkpoint (包含完整权重和结构)
        # 注意: PL checkpoint 包含 Lightning 结构，必须 weights_only=False
        checkpoint = torch.load(gen_ckpt_path, map_location="cpu", weights_only=False)
        
        # 处理 checkpoint 格式
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 清理键名前缀
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("model.", "").replace("mt5.", "")
                cleaned_state_dict[new_k] = v
            state_dict = cleaned_state_dict
        
        # 从 checkpoint 获取配置，直接创建完整模型
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            config = AutoConfig.from_dict(config_dict)
        else:
            # 使用默认配置
            config = AutoConfig.from_pretrained(REPO_ID)
        
        # 创建模型结构
        model_status.model_generator = MT5ForConditionalGeneration.from_config(config)
        model_status.model_generator.load_state_dict(state_dict, strict=False)
        model_status.model_generator.eval()
        
        # 直接从 checkpoint 目录加载 tokenizer（不单独下载 base model）
        try:
            model_status.tokenizer_generator = AutoTokenizer.from_pretrained(REPO_ID, legacy=False)
        except:
            # Fallback: 使用通用 mT5 tokenizer
            model_status.tokenizer_generator = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
        
        model_status.generator_loaded = True
        logger.info("生成模型加载成功!")
        
    except Exception as e:
        logger.error(f"生成模型加载失败: {e}")
        # Fallback: 直接从 HF 加载完整模型
        try:
            model_status.model_generator = MT5ForConditionalGeneration.from_pretrained(REPO_ID)
            model_status.tokenizer_generator = AutoTokenizer.from_pretrained(REPO_ID, legacy=False)
            model_status.model_generator.eval()
            model_status.generator_loaded = True
            logger.info("生成模型(Fallback)加载成功")
        except Exception as e2:
            logger.error(f"生成模型 Fallback 也失败: {e2}")
    
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
    "https://sy-s-web-3.pages.dev",  # Cloudflare Pages
    "http://localhost:5000",          # 本地开发
    "http://localhost:5173",           # Vite 开发服务器
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    snippet: str = Field(..., min_length=5, description="违规条款原文")
    violation_type: str = Field(..., description="违规类型ID，如 I1")

class UrlRequest(BaseModel):
    url: str = Field(..., description="目标URL")

# ==========================================
# 辅助函数
# ==========================================
def split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    sentences = re.split(r'[。；！？；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def roberta_predict(sentence: str) -> List[float]:
    """使用 BERT-MoE 分类模型预测"""
    if not model_status.classifier_loaded:
        logger.warning("分类模型未加载，返回默认概率")
        return [0.0] * 11
    
    inputs = model_status.tokenizer_classifier(
        sentence, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model_status.model_classifier(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    if not isinstance(probs, list):
        probs = [probs]
    
    return probs

def get_legal_basis_from_rag(violation_type: str, context: Optional[str] = None) -> str:
    """使用 RAG 检索获取法律依据"""
    if not RAG_AVAILABLE or retriever is None:
        # 回退到静态配置
        for name, info in INDICATORS.items():
            if info["id"] == violation_type:
                return info["legal_basis"]
        return "《个人信息保护法》"
    
    try:
        results = retriever.retrieve_by_violation_type(violation_type, context=context, top_k=3)
        if results:
            legal_refs = []
            for result in results[:2]:
                ref = f"{result.source} {result.article}"
                legal_refs.append(ref)
            return "；".join(legal_refs) if legal_refs else INDICATORS.get(
                ID_TO_INDICATOR.get(violation_type, ""), {}
            ).get("legal_basis", "《个人信息保护法》")
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
    all_violation_ids = set()
    
    for sentence in sentences:
        # BERT-MoE 分类预测
        probs = roberta_predict(sentence)
        
        # 11类 → 12类 映射
        if map_to_12_classes is not None:
            violation_ids = map_to_12_classes(probs, sentence)
        else:
            # Fallback: 使用概率最高的类别
            violation_ids = [f"I{max(0, i-1) + 1}" for i, p in enumerate(probs) if p > 0.5][:3]
        
        for v_id in violation_ids:
            indicator_name = ID_TO_INDICATOR.get(v_id)
            if indicator_name and v_id not in all_violation_ids:
                all_violation_ids.add(v_id)
                
                # RAG 获取法律依据
                legal_basis = get_legal_basis_from_rag(v_id, context=sentence)
                
                violations_list.append({
                    "indicator": indicator_name,
                    "violation_name": indicator_name,
                    "violation_id": v_id,
                    "snippet": sentence,
                    "legal_basis": legal_basis,
                    "confidence": round(probs[min(violation_ids.index(v_id), len(probs)-1)] if violation_ids else 0.5, 3)
                })
    
    # 计算合规评分
    score, risk_level = calculate_compliance_score(list(all_violation_ids))
    
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
        "created_at": project.created_at.isoformat()
    }

@app.post("/api/v1/rectify")
async def rectify_snippet(
    request: RectifyRequest,
    current_user: User = Depends(get_current_user)
):
    """生成违规条款的整改建议"""
    # RAG 检索相关法律条款
    legal_context = get_legal_basis_from_rag(request.violation_type, context=request.snippet)
    
    # 使用 mT5 生成整改建议
    if model_status.generator_loaded:
        prompt = f"""请根据以下法律规范，修改违规条款使其符合合规要求。

法律规范：{legal_context}

违规条款：{request.snippet}

整改后（保持原文风格，只修改违规内容）："""
        
        inputs = model_status.tokenizer_generator(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model_status.model_generator.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        suggested_text = model_status.tokenizer_generator.decode(outputs[0], skip_special_tokens=True)
    else:
        # Fallback: 返回通用建议
        indicator_name = ID_TO_INDICATOR.get(request.violation_type, "未知违规")
        suggested_text = f"建议修改条款内容，确保符合{indicator_name}的合规要求。"
    
    # 计算diff
    dmp = diff_match_patch()
    diffs = dmp.diff_main(request.snippet, suggested_text)
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
