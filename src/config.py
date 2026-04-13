"""
配置管理模块
集中管理系统配置，支持从环境变量和配置文件加载
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json
import os


@dataclass
class Config:
    """系统配置类"""
    
    # ==================== 路径配置 ====================
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    knowledge_dir: Path = field(init=False)
    vector_store_path: Path = field(init=False)
    
    # ==================== 模型配置 ====================
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    analyzer_model: str = "EnlightenedAI/TCSI_pp_zh"
    
    # ==================== 检索配置 ====================
    default_top_k: int = 5
    chunk_size: int = 300
    chunk_overlap: int = 50
    
    # ==================== 合规检查配置 ====================
    enabled_violation_types: List[str] = field(
        default_factory=lambda: [f"I{i}" for i in range(1, 13)]
    )
    
    def __post_init__(self):
        """初始化依赖路径"""
        self.knowledge_dir = self.project_root / "knowledge"
        self.vector_store_path = self.project_root / "vector_store"
        
        # 确保目录存在
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        config = cls()
        
        # 覆盖环境变量配置
        if embedding := os.getenv("EMBEDDING_MODEL"):
            config.embedding_model = embedding
        
        if analyzer := os.getenv("ANALYZER_MODEL"):
            config.analyzer_model = analyzer
        
        if top_k := os.getenv("DEFAULT_TOP_K"):
            config.default_top_k = int(top_k)
        
        return config
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "embedding_model": self.embedding_model,
            "analyzer_model": self.analyzer_model,
            "default_top_k": self.default_top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "enabled_violation_types": self.enabled_violation_types,
            "knowledge_dir": str(self.knowledge_dir),
            "vector_store_path": str(self.vector_store_path),
        }


# 全局配置实例
_default_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def reset_config() -> None:
    """重置配置（用于测试）"""
    global _default_config
    _default_config = None
