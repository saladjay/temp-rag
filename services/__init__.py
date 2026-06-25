"""服务层模块 - 提供各类外部服务的封装"""

# 向量服务
from .vector_service import VectorService, MockVectorService

# 知识库服务
from .kb_service import KnowledgeBaseService, MockKnowledgeBaseService, RetrievalMetrics

# LlamaIndex 服务
from .llamaindex_service import LlamaIndexService, MockLlamaIndexService

# 云端嵌入服务
from .cloud_embedding_service import CloudEmbeddingService, MockCloudEmbeddingService

# 云端重排序服务
from .cloud_rerank_service import CloudRerankService, MockCloudRerankService, RerankResult

# 云端补全服务
from .cloud_completion_service import CloudCompletionService, MockCloudCompletionService, CompletionResult

# MinerU 文档解析服务
from .mineru_service import MinerUService, MockMinerUService, MinerUResult

# 文档解析服务
from .document_parser_service import get_document_parser, DocumentParserService

# 问题生成服务
from .question_generator_service import get_question_generator, QuestionGeneratorService

# Milvus 知识库查询服务
from .milvus_knowledge_service import (
    MilvusKnowledgeClient,
    MockMilvusKnowledgeClient,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeResult,
    SearchType,
    FileType,
    query_knowledge,
)

__all__ = [
    # 向量服务
    "VectorService",
    "MockVectorService",
    # 知识库服务
    "KnowledgeBaseService",
    "MockKnowledgeBaseService",
    "RetrievalMetrics",
    # LlamaIndex 服务
    "LlamaIndexService",
    "MockLlamaIndexService",
    # 云端嵌入服务
    "CloudEmbeddingService",
    "MockCloudEmbeddingService",
    # 云端重排序服务
    "CloudRerankService",
    "MockCloudRerankService",
    "RerankResult",
    # 云端补全服务
    "CloudCompletionService",
    "MockCloudCompletionService",
    "CompletionResult",
    # MinerU 文档解析服务
    "MinerUService",
    "MockMinerUService",
    "MinerUResult",
    # 文档解析服务
    "get_document_parser",
    "DocumentParserService",
    # 问题生成服务
    "get_question_generator",
    "QuestionGeneratorService",
    # Milvus 知识库查询服务
    "MilvusKnowledgeClient",
    "MockMilvusKnowledgeClient",
    "KnowledgeQueryRequest",
    "KnowledgeQueryResponse",
    "KnowledgeResult",
    "SearchType",
    "FileType",
    "query_knowledge",
]