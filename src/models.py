"""
法律知识库数据模型
定义法律条款、违规映射等核心数据结构
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum


class RiskLevel(str, Enum):
    """风险等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ==================== 法律条款相关模型 ====================

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
    @classmethod
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


# ==================== 违规映射相关模型 ====================

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


# ==================== 知识库元数据 ====================

class KnowledgeBaseMeta(BaseModel):
    """知识库元数据"""
    version: str = Field(default="1.0.0", description="版本")
    last_updated: str = Field(..., description="最后更新时间")
    laws_count: int = Field(0, description="法律数量")
    total_articles: int = Field(0, description="总条款数")
    coverage: Dict[str, List[str]] = Field(default_factory=dict, description="违规类型覆盖")


# ==================== 检索结果 ====================

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
