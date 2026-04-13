"""
法律知识库加载器
从JSON文件加载法律条款，支持内置和自定义法律分离
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from src.models import (
    Article, 
    LawDocument, 
    ViolationMapping, 
    ViolationMappingConfig,
    KnowledgeBaseMeta,
    LawMeta,
    LawReference,
    ViolationExample,
    RiskLevel
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedKnowledge:
    """已加载的知识库数据"""
    articles: Dict[str, Article] = field(default_factory=dict)
    laws: Dict[str, LawDocument] = field(default_factory=dict)
    mapping: Dict[str, ViolationMapping] = field(default_factory=dict)
    violation_to_articles: Dict[str, List[str]] = field(default_factory=dict)
    article_count: int = 0
    law_count: int = 0


class LegalKBLoader:
    """
    法律知识库加载器
    
    从JSON文件加载法律条款，支持:
    - 内置法律: knowledge/laws/*.json
    - 自定义法律: knowledge/custom_laws/*.json
    - 违规映射: knowledge/mapping/*.json
    
    Usage:
        loader = LegalKBLoader()
        loader.load_all()
        articles = loader.get_articles_by_violation("I1")
    """
    
    def __init__(self, knowledge_dir: Optional[Path] = None):
        """
        Args:
            knowledge_dir: 知识库目录路径，默认使用项目根目录/knowledge
        """
        if knowledge_dir is None:
            from src.config import get_config
            knowledge_dir = get_config().knowledge_dir
        
        self.knowledge_dir = Path(knowledge_dir)
        self.laws_dir = self.knowledge_dir / "laws"
        self.custom_laws_dir = self.knowledge_dir / "custom_laws"
        self.mapping_dir = self.knowledge_dir / "mapping"
        
        self._knowledge: Optional[LoadedKnowledge] = None
    
    @property
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._knowledge is not None
    
    @property
    def knowledge(self) -> LoadedKnowledge:
        """获取已加载的知识库"""
        if self._knowledge is None:
            self.load_all()
        return self._knowledge
    
    def load_all(self) -> LoadedKnowledge:
        """
        加载所有知识库数据
        
        Returns:
            LoadedKnowledge: 已加载的知识库
        """
        self._knowledge = LoadedKnowledge()
        
        # 1. 加载内置法律
        self._load_laws_from_dir(self.laws_dir, is_custom=False)
        
        # 2. 加载自定义法律
        self._load_laws_from_dir(self.custom_laws_dir, is_custom=True)
        
        # 3. 加载违规映射
        self._load_mappings()
        
        # 4. 建立违规类型到条款的索引
        self._build_violation_index()
        
        logger.info(f"知识库加载完成: {self._knowledge.law_count} 部法律, "
                   f"{self._knowledge.article_count} 条条款")
        
        return self._knowledge
    
    def _load_laws_from_dir(self, directory: Path, is_custom: bool = False) -> None:
        """从目录加载法律文件"""
        if not directory.exists():
            logger.warning(f"法律目录不存在: {directory}")
            return
        
        for json_file in directory.glob("*.json"):
            try:
                law = self._load_single_law(json_file)
                if law:
                    law_name = law.law_meta.law_name
                    self._knowledge.laws[law_name] = law
                    self._knowledge.law_count += 1
                    
                    for article in law.articles:
                        self._knowledge.articles[article.article_id] = article
                        self._knowledge.article_count += 1
                    
                    if is_custom:
                        logger.info(f"加载自定义法律: {law_name} ({len(law.articles)} 条)")
                    else:
                        logger.debug(f"加载内置法律: {law_name} ({len(law.articles)} 条)")
                        
            except Exception as e:
                logger.error(f"加载法律文件失败 {json_file}: {e}")
    
    def _load_single_law(self, file_path: Path) -> Optional[LawDocument]:
        """加载单个法律文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理 law_meta 字段名映射
        if 'law_meta' not in data and '_meta' in data:
            data['law_meta'] = data.pop('_meta')
        
        # 构建 LawDocument
        meta = LawMeta(**data.get('law_meta', {}))
        
        articles = []
        for art_data in data.get('articles', []):
            try:
                article = Article(**art_data)
                articles.append(article)
            except Exception as e:
                logger.warning(f"解析条款失败: {art_data.get('article_id', 'unknown')}, {e}")
        
        return LawDocument(law_meta=meta, articles=articles)
    
    def _load_mappings(self) -> None:
        """加载违规映射配置"""
        mapping_file = self.mapping_dir / "violation_mapping.json"
        
        if not mapping_file.exists():
            logger.warning(f"违规映射文件不存在: {mapping_file}")
            return
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = ViolationMappingConfig(**data)
        self._knowledge.mapping = config.mappings
        logger.info(f"加载违规映射: {len(config.mappings)} 种违规类型")
    
    def _build_violation_index(self) -> None:
        """构建违规类型到条款的索引"""
        for article_id, article in self._knowledge.articles.items():
            for vt in article.violation_types:
                if vt not in self._knowledge.violation_to_articles:
                    self._knowledge.violation_to_articles[vt] = []
                self._knowledge.violation_to_articles[vt].append(article_id)
    
    # ==================== 公共接口 ====================
    
    def get_article(self, article_id: str) -> Optional[Article]:
        """根据条款ID获取法律条款"""
        return self._knowledge.articles.get(article_id)
    
    def get_law(self, law_name: str) -> Optional[LawDocument]:
        """根据法律名称获取法律文档"""
        return self._knowledge.laws.get(law_name)
    
    def get_articles_by_violation(self, violation_type: str) -> List[Article]:
        """
        获取指定违规类型关联的所有法律条款
        
        Args:
            violation_type: 违规类型，如 "I1", "I2"
            
        Returns:
            关联的法律条款列表
        """
        article_ids = self._knowledge.violation_to_articles.get(violation_type, [])
        return [self._knowledge.articles[aid] for aid in article_ids 
                if aid in self._knowledge.articles]
    
    def get_all_violation_types(self) -> Set[str]:
        """获取所有已定义的违规类型"""
        return set(self._knowledge.violation_to_articles.keys())
    
    def search_articles(self, query: str, limit: int = 10) -> List[Article]:
        """
        根据关键词搜索相关条款
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的法律条款列表
        """
        query_lower = query.lower()
        results = []
        
        for article in self._knowledge.articles.values():
            # 匹配标题、内容或关键词
            score = 0
            if query_lower in article.title.lower():
                score += 3
            if query_lower in article.content.lower():
                score += 2
            for kw in article.keywords:
                if query_lower in kw.lower():
                    score += 1
            
            if score > 0:
                results.append((score, article))
        
        # 按分数排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [art for _, art in results[:limit]]
    
    def get_coverage_summary(self) -> KnowledgeBaseMeta:
        """获取知识库覆盖摘要"""
        return KnowledgeBaseMeta(
            version="1.0.0",
            last_updated=self._get_latest_update(),
            laws_count=len(self._knowledge.laws),
            total_articles=self._knowledge.article_count,
            coverage=self._knowledge.violation_to_articles
        )
    
    def _get_latest_update(self) -> str:
        """获取最新更新时间"""
        latest = None
        for law in self._knowledge.laws.values():
            date = law.law_meta.effective_date
            if latest is None or date > latest:
                latest = date
        return latest or "2024-01-01"
    
    def reload(self) -> LoadedKnowledge:
        """重新加载知识库"""
        self._knowledge = None
        return self.load_all()
