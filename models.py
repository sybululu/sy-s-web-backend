"""
数据模型模块
合并了数据库模型和知识库模型
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# ==========================================
# SQLAlchemy 数据库模型
# ==========================================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    name = Column(String(200), nullable=False)
    source_type = Column(String(20))
    
    score = Column(Float)
    risk_level = Column(String(20))
    
    result_json = Column(Text)
    raw_text = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User", back_populates="projects")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./privacy_checker.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# Pydantic API 模型
# ==========================================

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    source_type: Optional[str] = "text"
    
    @field_validator('text')
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

class UrlRequest(BaseModel):
    url: str

# ==========================================
# 知识库模型 (来自 src.models)
# ==========================================

class RiskLevel(str, Enum):
    """风险等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ArticleAnnotation(BaseModel):
    """条款注释"""
    key_points: List[str] = Field(default_factory=list, description="要点")
    common_issues: List[str] = Field(default_factory=list, description="常见问题")


class Article(BaseModel):
    """法律条款"""
    article_id: str = Field(..., description="条款唯一标识，如 PIPL-017")
    article_number: str = Field(..., description="条款编号，如 第十七条")
    title: str = Field(..., description="条款标题")
    content: str = Field(..., description="条款原文内容")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    violation_types: List[str] = Field(
        default_factory=list, 
        description="关联的违规类型，如 I1, I2"
    )
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="风险等级")
    annotations: Optional[ArticleAnnotation] = Field(None, description="条款注释")
    
    @field_validator('violation_types')
    def validate_violation_types(cls, v):
        """验证违规类型格式"""
        for vt in v:
            if not vt.startswith('I'):
                raise ValueError(f"违规类型必须以 I 开头: {vt}")
        return v


class LawMeta(BaseModel):
    """法律元数据"""
    law_name: str = Field(..., description="法律名称")
    law_name_en: Optional[str] = Field(None, description="英文名称")
    effective_date: str = Field(..., description="生效日期 YYYY-MM-DD")
    jurisdiction: str = Field(default="中国", description="管辖区域")
    version: str = Field(default="1.0", description="版本号")


class LawDocument(BaseModel):
    """法律文档"""
    law_meta: LawMeta = Field(..., alias="law_meta")
    articles: List[Article] = Field(default_factory=list, description="条款列表")
    
    class Config:
        populate_by_name = True


class LawReference(BaseModel):
    """法律引用"""
    law: str = Field(..., description="法律简称，如 PIPL")
    article: str = Field(..., description="条款编号，如 第六条")


class ViolationExample(BaseModel):
    """违规示例"""
    violation: str = Field(..., description="违规示例")
    compliant: str = Field(..., description="合规示例")


class ViolationMapping(BaseModel):
    """违规类型映射"""
    name: str = Field(..., description="违规类型名称")
    category: str = Field(..., description="所属类别")
    primary_laws: List[LawReference] = Field(default_factory=list, description="主要涉及法律")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    examples: Optional[ViolationExample] = Field(None, description="示例")


class ViolationMappingConfig(BaseModel):
    """违规映射配置"""
    mappings: Dict[str, ViolationMapping] = Field(..., description="映射字典")


class KnowledgeBaseMeta(BaseModel):
    """知识库元数据"""
    version: str = Field(default="1.0.0", description="版本")
    last_updated: str = Field(..., description="最后更新时间")
    laws_count: int = Field(0, description="法律数量")
    total_articles: int = Field(0, description="总条款数")
    coverage: Dict[str, List[str]] = Field(default_factory=dict, description="违规类型覆盖")


class SearchResult(BaseModel):
    """检索结果"""
    article_id: str = Field(..., description="条款ID")
    law: str = Field(..., description="法律名称")
    article_number: str = Field(..., description="条款编号")
    title: str = Field(..., description="条款标题")
    content: str = Field(..., description="条款内容")
    relevance_score: float = Field(..., description="相关度分数 0-1")
    keywords_matched: List[str] = Field(default_factory=list, description="匹配的关键词")
    violation_types: List[str] = Field(default_factory=list, description="关联的违规类型")
    law_reference: Optional[str] = Field(None, description="完整法律引用")


class RetrievedChunk(BaseModel):
    """检索到的文本块"""
    text: str = Field(..., description="文本内容")
    metadata: Dict = Field(default_factory=dict, description="元数据")
    source: str = Field(..., description="来源")
    score: float = Field(0.0, description="相似度分数")
