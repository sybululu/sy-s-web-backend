"""
法律条款检索器
整合知识库加载和向量检索，提供高级检索接口
"""
import logging
from typing import List, Optional, Dict

from src.loader import LegalKBLoader
from src.store import VectorStore
from src.models import Article, SearchResult, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """
    法律条款检索器
    
    功能:
    - 根据违规类型检索相关法律条款
    - 通用语义检索
    - 支持过滤和排序
    """
    
    def __init__(
        self,
        loader: Optional[LegalKBLoader] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Args:
            loader: 知识库加载器
            vector_store: 向量存储
        """
        self.loader = loader or LegalKBLoader()
        self.vector_store = vector_store or VectorStore()
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化检索器，加载知识库和向量库"""
        if self._initialized:
            return
        
        # 确保知识库已加载
        self.loader.load_all()
        
        # 尝试恢复向量库
        if not self.vector_store.restore():
            # 如果恢复失败，从知识库初始化
            articles = list(self.loader.knowledge.articles.values())
            if articles:
                self.vector_store.initialize(articles)
                self.vector_store.persist()
        
        self._initialized = True
        logger.info("检索器初始化完成")
    
    def retrieve_by_violation_type(
        self,
        violation_type: str,
        context: Optional[str] = None,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        根据违规类型检索相关法律条款
        
        Args:
            violation_type: 违规类型，如 "I1", "I2"
            context: 违规上下文描述
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        self.initialize()
        
        # 1. 先获取该违规类型关联的条款
        related_articles = self.loader.get_articles_by_violation(violation_type)
        
        if not related_articles:
            logger.warning(f"未找到违规类型 {violation_type} 的映射条款")
            return []
        
        # 2. 使用向量检索获取语义相似的结果
        search_results = self.vector_store.search_by_violation(
            violation_type, 
            context=context,
            top_k=top_k
        )
        
        # 3. 合并结果
        results = []
        seen_ids = set()
        
        # 优先添加映射的条款
        for article in related_articles:
            results.append(self._article_to_search_result(article))
            seen_ids.add(article.article_id)
        
        # 添加向量检索结果
        for chunk in search_results:
            article_id = chunk.metadata.get("article_id")
            if article_id and article_id not in seen_ids:
                article = self.loader.get_article(article_id)
                if article:
                    results.append(self._article_to_search_result(article, chunk.score))
                    seen_ids.add(article_id)
            
            if len(results) >= top_k + len(related_articles):
                break
        
        return results[:top_k]
    
    def retrieve_by_query(
        self,
        query: str,
        violation_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        通用语义检索
        
        Args:
            query: 查询文本
            violation_type: 可选的违规类型过滤
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        self.initialize()
        
        filters = None
        if violation_type:
            filters = {"violation_types": [violation_type]}
        
        chunks = self.vector_store.search(query, top_k=top_k, filters=filters)
        
        results = []
        for chunk in chunks:
            article_id = chunk.metadata.get("article_id")
            article = self.loader.get_article(article_id)
            if article:
                results.append(self._article_to_search_result(article, chunk.score))
        
        return results
    
    def _article_to_search_result(
        self, 
        article: Article, 
        relevance_score: Optional[float] = None
    ) -> SearchResult:
        """将Article转换为SearchResult"""
        # 获取法律名称
        article_id_prefix = article.article_id.split('-')[0]
        law_map = {
            "PIPL": "个人信息保护法",
            "DSL": "数据安全法",
            "CSL": "网络安全法",
            "GB": "GB/T 35273-2020"
        }
        law_name = law_map.get(article_id_prefix, article_id_prefix)
        
        return SearchResult(
            article_id=article.article_id,
            law=law_name,
            article_number=article.article_number,
            title=article.title,
            content=article.content,
            relevance_score=relevance_score or 1.0,
            keywords_matched=article.keywords,
            violation_types=article.violation_types,
            law_reference=f"《{law_name}》{article.article_number}"
        )
    
    def get_violation_mapping(self, violation_type: str) -> Optional[Dict]:
        """获取违规类型映射详情"""
        return self.loader.knowledge.mapping.get(violation_type)
    
    def get_coverage_info(self) -> Dict:
        """获取知识库覆盖信息"""
        summary = self.loader.get_coverage_summary()
        return {
            "version": summary.version,
            "laws_count": summary.laws_count,
            "total_articles": summary.total_articles,
            "violation_coverage": {
                vt: len(articles) 
                for vt, articles in summary.coverage.items()
            },
            "missing_violations": [
                f"I{i}" for i in range(1, 13) 
                if f"I{i}" not in summary.coverage
            ]
        }
