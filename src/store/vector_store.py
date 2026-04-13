"""
向量存储模块
基于FAISS实现，支持持久化和法律条款检索
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from src.models import Article, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS向量存储
    
    支持:
    - 从Article列表初始化
    - 持久化到磁盘
    - 从磁盘恢复
    - 语义检索
    """
    
    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        persist_path: Optional[Path] = None
    ):
        """
        Args:
            embedding_model: 嵌入模型名称
            persist_path: 持久化路径
        """
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexIDMap] = None
        self.chunks: List[Dict] = []
        self.persist_path = persist_path or Path("./vector_store")
        
        # 索引状态
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized and self.index is not None
    
    def _load_embedding_model(self) -> None:
        """延迟加载嵌入模型"""
        if self.embedding_model is None:
            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("嵌入模型加载完成")
    
    def initialize(self, articles: List[Article]) -> None:
        """
        从Article列表初始化向量库
        
        Args:
            articles: 法律条款列表
        """
        self._load_embedding_model()
        
        # 准备文本
        self.chunks = []
        for article in articles:
            self.chunks.append({
                "text": article.content,
                "article_id": article.article_id,
                "title": article.title,
                "article_number": article.article_number,
                "keywords": article.keywords,
                "violation_types": article.violation_types,
                "law": article.article_id.split('-')[0] if '-' in article.article_id else ""
            })
        
        if not self.chunks:
            logger.warning("没有条款需要索引")
            return
        
        # 生成嵌入
        texts = [chunk["text"] for chunk in self.chunks]
        logger.info(f"正在为 {len(texts)} 个条款生成嵌入...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # 归一化用于余弦相似度
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        
        # 使用文档索引作为ID
        ids = np.array(range(len(self.chunks))).astype('int64')
        self.index.add_with_ids(embeddings, ids)
        
        self._initialized = True
        logger.info(f"向量库初始化完成: {self.index.ntotal} 个向量")
    
    def persist(self) -> None:
        """保存向量库到磁盘"""
        if not self.is_initialized:
            logger.warning("向量库未初始化，无法持久化")
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        index_path = self.persist_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # 保存chunks元数据
        chunks_path = self.persist_path / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"向量库已持久化到: {self.persist_path}")
    
    def restore(self) -> bool:
        """
        从磁盘恢复向量库
        
        Returns:
            是否恢复成功
        """
        index_path = self.persist_path / "index.faiss"
        chunks_path = self.persist_path / "chunks.pkl"
        
        if not index_path.exists() or not chunks_path.exists():
            logger.warning("持久化文件不存在")
            return False
        
        try:
            self._load_embedding_model()
            
            # 恢复索引
            self.index = faiss.read_index(str(index_path))
            
            # 恢复chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            self._initialized = True
            logger.info(f"向量库已恢复: {len(self.chunks)} 个条款")
            return True
            
        except Exception as e:
            logger.error(f"恢复向量库失败: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[RetrievedChunk]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件 (violation_types, law)
            
        Returns:
            检索结果列表
        """
        if not self.is_initialized:
            logger.error("向量库未初始化")
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        search_k = min(top_k * 2, len(self.chunks))  # 多搜索一些以便过滤
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            chunk = self.chunks[idx]
            
            # 应用过滤
            if filters:
                if "violation_types" in filters:
                    if not any(vt in chunk.get("violation_types", []) 
                              for vt in filters["violation_types"]):
                        continue
                if "law" in filters:
                    if chunk.get("law") != filters["law"]:
                        continue
            
            results.append(RetrievedChunk(
                text=chunk["text"],
                metadata={
                    "article_id": chunk["article_id"],
                    "title": chunk["title"],
                    "keywords": chunk["keywords"]
                },
                source=chunk["article_id"],
                score=float(score)
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_violation(
        self, 
        violation_type: str, 
        context: Optional[str] = None,
        top_k: int = 3
    ) -> List[RetrievedChunk]:
        """
        根据违规类型检索相关条款
        
        Args:
            violation_type: 违规类型，如 "I1"
            context: 可选的上下文描述
            top_k: 返回数量
            
        Returns:
            检索结果
        """
        # 先按违规类型过滤
        filters = {"violation_types": [violation_type]}
        
        if context:
            # 结合上下文语义检索
            return self.search(context, top_k=top_k, filters=filters)
        else:
            # 仅使用违规类型关键词
            return self.search(f"隐私政策 {violation_type} 合规", 
                             top_k=top_k, filters=filters)
