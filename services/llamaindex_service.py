"""llamaindex 服务模块 - 管理向量索引和检索"""
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.models.document import Document, SearchResult, SearchRequest
from app.config import settings


# TODO: 当 llamaindex 正式集成后启用
# 目前使用 Mock 实现来测试框架

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.readers import SimpleDirectoryReader
    from llama_index.core.settings import Settings as LlamaIndexSettings
    from llama_index.core.query_engine import QueryBundle
    from llama_index.core import VectorStoreQueryMode
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False


class LlamaIndexService:
    """llamaindex 服务类"""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        use_llama_cloud: bool = False
    ):
        """初始化 llamaindex 服务

        Args:
            persist_dir: 持久化存储目录
            use_llama_cloud: 是否使用 Llama Cloud
        """
        self.use_llama_cloud = use_llama_cloud
        self._index: Optional[List[Document]] = None
        self._persist_dir = persist_dir or "./data/llamaindex"

        # 确保 persist 目录存在
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

    def _check_available(self) -> None:
        """检查 llamaindex 是否可用"""
        if not LLAMA_INDEX_AVAILABLE:
            raise RuntimeError(
                "llamaindex is not installed. Install with: uv sync --extra vector"
            )

    def initialize(self, documents: List[Document]) -> None:
        """初始化索引并添加文档"""
        self._check_available()

        # TODO: 替换为实际的嵌入模型或使用本地模型
        # 这里使用简单的示例实现

        # 准备文档节点
        llama_documents = []
        for doc in documents:
            llama_documents.append({
                "text": doc.content,
                "metadata": {
                    "id": doc.id,
                    "title": doc.title,
                    **doc.metadata
                }
            })

        # 创建索引（Mock实现）
        self._index = documents

        # 持久化
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """执行搜索"""
        if self._index is None:
            return []

        results = []
        for doc in self._index:
            # 应用元数据过滤
            if filter_dict:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_dict.items()
                )
                if not match:
                    continue

            # 计算相似度（简单文本匹配作为Mock）
            # 在真实实现中，这里会使用 llamaindex 的向量搜索
            score = 1.0 if query.lower() in doc.content.lower() else 0.0

            # 应用最小分数过滤
            if score < min_score:
                continue

            results.append(SearchResult(document=doc, score=score))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def add_document(self, document: Document) -> str:
        """添加单个文档到索引"""
        return self.add_documents([document])[0] if document.id else ""

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到索引"""
        doc_ids = []
        for doc in documents:
            # 生成 ID（如果没有）
            if not doc.id:
                import uuid
                doc.id = str(uuid.uuid4())

            doc_ids.append(doc.id)

        # 如果索引已初始化，则增量添加
        if self._index is not None:
            self._index.extend(documents)
        else:
            # 初始化索引
            self.initialize(documents)

        return doc_ids

    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        if self._index is None:
            return None

        for doc in self._index:
            if doc.id == doc_id:
                return doc
        return None

    def clear(self) -> None:
        """清空索引"""
        # 删除索引存储
        import shutil
        if Path(self._persist_dir).exists():
            shutil.rmtree(self._persist_dir)
        self._index = None

    def count(self) -> int:
        """返回文档数量"""
        if self._index is None:
            return 0
        return len(self._index)

    def from_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_exts: Optional[List[str]] = None
    ) -> List[Document]:
        """从目录加载文档"""
        # 简化实现，直接读取目录
        import os
        from pathlib import Path as PathLib

        doc_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc_list.append(Document(
                                content=content,
                                metadata={
                                    "file_name": file,
                                    "file_path": os.path.join(root, file)
                                }
                            ))
                    except Exception:
                        pass

        return doc_list


class MockLlamaIndexService(LlamaIndexService):
    """Mock llamaindex 服务，用于测试"""

    def __init__(self):
        """初始化 Mock 服务"""
        super().__init__(persist_dir="./data/mock_llamaindex")
        self._mock_documents: Dict[str, Document] = {}

    def initialize(self, documents: List[Document]) -> None:
        """初始化索引并添加文档"""
        for doc in documents:
            if doc.id:
                self._mock_documents[doc.id] = doc

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """执行搜索"""
        from app.services.vector_service import MockVectorService
        import numpy as np

        vector_service = MockVectorService(dimension=384, fixed_seed=42)
        query_embedding = vector_service.encode(query)

        results = []
        for doc in list(self._mock_documents.values()):
            # 应用元数据过滤
            if filter_dict:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_dict.items()
                )
                if not match:
                    continue

            # 计算相似度
            if doc.embedding:
                doc_vec = np.array(doc.embedding)
                score = vector_service.similarity(query_embedding, doc_vec)
            else:
                # 如果没有嵌入，使用基于文本的简单匹配
                score = 1.0 if query.lower() in doc.content.lower() else 0.0

            # 应用最小分数过滤
            if score < min_score:
                continue

            results.append(SearchResult(document=doc, score=score))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def add_document(self, document: Document) -> str:
        """添加文档"""
        if not document.id:
            import uuid
            document.id = str(uuid.uuid4())

        self._mock_documents[document.id] = document
        return document.id

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档"""
        doc_ids = []
        for doc in documents:
            if not doc.id:
                import uuid
                doc.id = str(uuid.uuid4())
            self._mock_documents[doc.id] = doc
            doc_ids.append(doc.id)
        return doc_ids

    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self._mock_documents.get(doc_id)

    def clear(self) -> None:
        """清空索引"""
        self._mock_documents.clear()

    def count(self) -> int:
        """返回文档数量"""
        return len(self._mock_documents)
