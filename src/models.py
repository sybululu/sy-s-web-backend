"""
RAG 模块专用的数据模型

定义法律知识库所需的 Pydantic 数据模型。
从根目录 models.py 导入并 re-export，避免循环依赖。
"""
import sys
from pathlib import Path

# 确保项目根目录可被找到
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 从根目录 models 导入知识库模型（仅 Pydantic 部分，不含 SQLAlchemy）
from models import (                          # noqa: E402
    Article,
    LawDocument,
    ViolationMapping,
    ViolationMappingConfig,
    KnowledgeBaseMeta,
    LawMeta,
    LawReference,
    ViolationExample,
    RiskLevel,
    SearchResult,
    RetrievedChunk,
)

__all__ = [
    "Article",
    "LawDocument",
    "ViolationMapping",
    "ViolationMappingConfig",
    "KnowledgeBaseMeta",
    "LawMeta",
    "LawReference",
    "ViolationExample",
    "RiskLevel",
    "SearchResult",
    "RetrievedChunk",
]
