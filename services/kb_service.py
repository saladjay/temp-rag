"""知识库服务模块 - 管理文档存储和检索"""
from typing import List, Optional, Dict, Any
import time
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from app.models.document import Document, SearchResult, SearchRequest
from app.services.vector_service import VectorService


@dataclass
class RetrievalMetrics:
    """检索指标"""
    recall_at_k: float
    precision_at_k: float
    mrr: float  # Mean Reciprocal Rank


class KnowledgeBaseService:
    """知识库服务类"""

    def __init__(self, vector_service: VectorService):
        """初始化知识库服务

        Args:
            vector_service: 向量服务实例
        """
        self.vector_service = vector_service
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add_document(self, document: Document) -> str:
        """添加文档到知识库

        Args:
            document: 文档对象

        Returns:
            文档ID
        """
        # 生成ID（如果没有）
        if not document.id:
            document.id = f"doc_{len(self._documents) + 1}"

        # 生成向量嵌入
        embedding = self.vector_service.encode(document.content)
        document.embedding = embedding.flatten().tolist()

        # 存储文档和嵌入
        self._documents[document.id] = document
        self._embeddings[document.id] = document.embedding

        return document.id

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档

        Args:
            documents: 文档列表

        Returns:
            文档ID列表
        """
        doc_ids = []
        for doc in documents:
            doc_ids.append(self.add_document(doc))
        return doc_ids

    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档

        Args:
            doc_id: 文档ID

        Returns:
            文档对象，不存在返回None
        """
        return self._documents.get(doc_id)

    def search(self, request: SearchRequest) -> SearchResult:
        """执行搜索

        Args:
            request: 搜索请求

        Returns:
            搜索结果
        """
        start_time = time.time()

        # 编码查询
        query_embedding = self.vector_service.encode(request.query).flatten()

        # 准备文档向量
        doc_ids = list(self._embeddings.keys())
        if not doc_ids:
            return SearchResult(
                document=Document(content=""),
                score=0.0,
            )

        doc_vectors = np.array([self._embeddings[doc_id] for doc_id in doc_ids])

        # 计算相似度
        similarities = self.vector_service.batch_similarity(query_embedding, doc_vectors)

        # 过滤元数据（如果有过滤条件）
        if request.filter:
            filtered_indices = [
                i for i, doc_id in enumerate(doc_ids)
                if self._matches_filter(doc_id, request.filter)
            ]
            if filtered_indices:
                doc_ids = [doc_ids[i] for i in filtered_indices]
                similarities = similarities[filtered_indices]
            else:
                return SearchResult(
                    document=Document(content=""),
                    score=0.0,
                )

        # 排序并取top_k
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        top_indices = [i for i in top_indices if similarities[i] >= request.min_score]

        # 构建结果
        results = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            score = float(similarities[idx])
            results.append(
                SearchResult(
                    document=self._documents[doc_id],
                    score=score
                )
            )

        execution_time = (time.time() - start_time) * 1000

        # 返回单个结果（兼容性）
        if results:
            return results[0]

        return SearchResult(document=Document(content=""), score=0.0)

    def search_all(self, request: SearchRequest) -> tuple[List[SearchResult], float]:
        """执行搜索并返回所有结果和执行时间

        Args:
            request: 搜索请求

        Returns:
            (结果列表, 执行时间ms)
        """
        start_time = time.time()

        # 编码查询
        query_embedding = self.vector_service.encode(request.query).flatten()

        # 准备文档向量
        doc_ids = list(self._embeddings.keys())
        if not doc_ids:
            return [], (time.time() - start_time) * 1000

        doc_vectors = np.array([self._embeddings[doc_id] for doc_id in doc_ids])

        # 计算相似度
        similarities = self.vector_service.batch_similarity(query_embedding, doc_vectors)

        # 过滤元数据（如果有过滤条件）
        if request.filter:
            filtered_indices = [
                i for i, doc_id in enumerate(doc_ids)
                if self._matches_filter(doc_id, request.filter)
            ]
            if filtered_indices:
                doc_ids = [doc_ids[i] for i in filtered_indices]
                similarities = similarities[filtered_indices]
            else:
                return [], (time.time() - start_time) * 1000

        # 排序并取top_k
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        top_indices = [i for i in top_indices if similarities[i] >= request.min_score]

        # 构建结果
        results = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            score = float(similarities[idx])
            results.append(
                SearchResult(
                    document=self._documents[doc_id],
                    score=score
                )
            )

        execution_time = (time.time() - start_time) * 1000
        return results, execution_time

    def _matches_filter(self, doc_id: str, filter_dict: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤条件

        Args:
            doc_id: 文档ID
            filter_dict: 过滤条件字典

        Returns:
            是否匹配
        """
        doc = self._documents.get(doc_id)
        if not doc:
            return False

        for key, value in filter_dict.items():
            if doc.metadata.get(key) != value:
                return False
        return True

    def clear(self):
        """清空知识库"""
        self._documents.clear()
        self._embeddings.clear()

    def count(self) -> int:
        """返回文档数量"""
        return len(self._documents)


class MockKnowledgeBaseService(KnowledgeBaseService):
    """Mock知识库服务，用于测试"""

    def __init__(self):
        """初始化Mock服务，使用Mock向量服务"""
        from app.services.vector_service import MockVectorService
        super().__init__(MockVectorService())
