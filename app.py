"""
隐私政策合规审查 API
整合了 RAG 架构的法律知识库检索

整改生成模型支持两种模式（通过环境变量 LLM_MODE 切换）:
  - "local" : 本地 GGUF (llama-cpp-python + Phi-4 Mini Q6_K)
  - "api"   : HuggingFace Inference API (默认，无需本地编译)
"""
from __future__ import annotations

import json
import os
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, InferenceClient

from models import User, Project, get_db, init_db, Article, RetrievedChunk
from auth import router as auth_router, get_current_user

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# LLM 模式配置
# ==========================================
LLM_MODE = os.getenv("LLM_MODE", "api")  # "local" 或 "api"
HF_TOKEN = os.getenv("HF_TOKEN", "")     # HF API Token（免费额度即可）
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "microsoft/Phi-4-mini-instruct")

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
# 模型加载 (HuggingFace Hub)
# ==========================================
print("正在从 HuggingFace Hub 加载模型，首次运行需要下载...")

# 1. 加载 RoBERTa 风险分类模型（基础模型 + 微调 checkpoint）
tokenizer_roberta = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model_roberta = AutoModelForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext", num_labels=12
)
# 从 HF Hub 下载微调 checkpoint 并加载
roberta_ckpt_path = hf_hub_download(
    repo_id="sybululu/bert-moe",
    filename="multi_classification_bertmoe.ckpt"
)
state_dict = torch.load(roberta_ckpt_path, map_location="cpu", weights_only=True)
model_roberta.load_state_dict(state_dict, strict=False)
model_roberta.eval()
print("RoBERTa 分类模型加载完成")

# 2. 加载整改生成模型（支持 local / api 双模式）
llm = None  # type: ignore

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
        logger.warning(f"本地 GGUF 模式加载失败，回退到 API 模式: {e}")
        LLM_MODE = "api"

if LLM_MODE == "api":
    # API 模式：使用 HuggingFace Inference API（无需本地编译，部署极快）
    llm = InferenceClient(model=LLM_MODEL_ID, token=HF_TOKEN or None)
    print(f"✅ 整改生成模型: HF Inference API 模式 (model={LLM_MODEL_ID})")

# ==========================================
# RAG 组件初始化
# ==========================================
# 使用 Any 类型注解，避免 RAG 模块导入失败时 NameError
legal_kb_loader: Optional[Any] = None
vector_store: Optional[Any] = None
retriever: Optional[Any] = None

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

# 违规类型整改提示语（与 INDICATORS 一一对应，按 ID 索引）
VIOLATION_HINTS = {
    "I1": "【重要】涉及收集个人信息，必须遵守最小必要原则，只能收集与服务直接相关的个人信息，禁止收集与服务无关的敏感信息。",
    "I2": "【重要】必须明确说明每项个人信息收集的具体目的和用途，不能使用模糊表述。",
    "I3": "【重要】涉及处理个人信息必须获得用户明确、知情、自愿的同意，不能捆绑授权。",
    "I4": "【重要】收集范围不得超过实现处理目的的最小必要范围。",
    "I5": "【重要】向第三方共享时必须明确说明接收方类型、共享目的、数据类型，禁止无限制共享。",
    "I6": "【重要】向第三方提供个人信息必须单独取得用户明示同意。",
    "I7": "【重要】必须明确说明第三方使用数据的目的和范围。",
    "I8": "【重要】必须明确数据存储期限，期限届满应予以删除或匿名化。",
    "I9": "【重要】必须说明数据销毁机制，承诺在约定保存期限届满后主动删除或匿名化处理。",
    "I10": "【重要】必须明确列举用户享有的各项权利及行使方式。",
    "I11": "【重要】必须提供便捷的渠道供用户行使权利，渠道必须易于发现和操作。",
    "I12": "【重要】必须明确权利响应时限，承诺在法定期限内处理用户请求。",
}

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
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'[。；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def roberta_predict(sentence: str) -> Dict[str, float]:
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
                if not any(v["indicator"] == indicator for v in violations_list):
                    # 获取违规类型ID
                    violation_id = INDICATORS[indicator]["id"]
                    # 使用 RAG 获取法律依据
                    legal_basis = get_legal_basis_from_rag(violation_id, context=sentence)
                    
                    violations_list.append({
                        "indicator": indicator,
                        "violation_id": violation_id,
                        "snippet": sentence,
                        "legal_basis": legal_basis
                    })

    penalty = sum(INDICATORS[ind]["weight"] * vi for ind, vi in violation_flags.items())
    total_score = round(max(0.0, 100.0 - (penalty * 100.0)), 1)

    if total_score >= SCORE_THRESHOLD_LOW_RISK:
        risk_level = "低风险"
    elif SCORE_THRESHOLD_HIGH_RISK <= total_score < SCORE_THRESHOLD_LOW_RISK:
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
    """整改违规条款"""
    # 使用 RAG 检索相关法律条款（失败时依次降级：前端传入值 → 静态配置）
    legal_context = get_legal_basis_from_rag(request.violation_type, context=request.original_snippet)
    if not legal_context or legal_context == "《个人信息保护法》":
        legal_context = request.legal_basis or legal_context

    # 获取整改提示语（模块级常量）
    violation_hint = VIOLATION_HINTS.get(request.violation_type, "【重要】必须符合《个人信息保护法》相关要求。")

    # 根据 mode 构建差异化 prompt + chat messages
    if request.mode == "summary":
        # 摘要模式：通俗解读违规原因与法律要求
        user_content = f"""请对以下隐私政策条款进行通俗易懂的解读，帮助非法律专业人士理解该条款存在的问题。

【法律依据】
{legal_context}

【违规类型说明】
{violation_hint}

【原条款】
{request.original_snippet}

请用简洁明了的语言回答以下三个问题：
1. 这条条款说了什么？（用一句话概括原意）
2. 它违反了什么法律规定？具体哪里不合规？
3. 合规版本应该包含哪些关键要素？"""
        system_content = "你是一位隐私政策合规解读专家，擅长将法律条文翻译成普通用户能理解的语言。请直接给出解读内容，不要添加多余的前缀或格式标记。"
    else:
        # 改写模式：以检测到的具体法律条文为依据，逐条改写
        indicator_name = ID_TO_INDICATOR.get(request.violation_type, "")
        indicator_legal = INDICATORS.get(indicator_name, {}).get("legal_basis", legal_context)

        user_content = f"""你是一位隐私政策合规专家。以下条款被检测出违反特定法律规定，请根据该法律的具体要求进行改写。

【检测结果】
- 违规类别：{indicator_name}（{request.violation_type}）
- 违反的法律条文：{indicator_legal}

【相关法律依据详情】
{legal_context}

【具体整改要求】
{violation_hint}

【原条款】
{request.original_snippet}

请严格参照上述法律条文的具体要求，将原条款改写为符合该法律规定的合规版本。
改写后的条款应当：
1. 直接回应上述法律条文的要求，补全缺失的法定要素
2. 保持隐私政策的专业表述风格，语言清晰准确
3. 不改变原条款的核心业务意图，仅修正不合规之处"""
        system_content = "你是一位隐私政策合规专家。请直接输出改写后的完整合规条款文本，不要添加任何解释、标注或前缀。"

    # 调用 LLM 生成整改建议（兼容 local / api 双模式)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    if LLM_MODE == "local" and hasattr(llm, 'create_chat_completion'):
        # 本地 GGUF 模式
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        suggested_text = response["choices"][0]["message"]["content"].strip()
    else:
        # HF Inference API 模式
        response = llm.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        suggested_text = response.choices[0].message.content.strip()
    
    return {
        "suggested_text": suggested_text,
        "legal_basis": legal_context,
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
