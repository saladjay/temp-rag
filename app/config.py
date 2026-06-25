"""Application configuration module.

This module provides configuration for the HTTP server and cloud services.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """Application configuration class for HTTP server and services."""

    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # 向量数据库配置
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "test-knowledge-base"
    pinecone_dimension: int = 384

    # 嵌入模型配置
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 云端服务通用认证令牌
    cloud_auth_token: Optional[str] = None

    # 云端嵌入服务配置
    cloud_embedding_url: Optional[str] = None
    cloud_embedding_model: str = "bge-m3"
    cloud_embedding_timeout: int = 30

    # 云端重排序服务配置
    cloud_rerank_url: Optional[str] = None
    cloud_rerank_model: str = "embed_rerank"
    cloud_rerank_timeout: int = 30

    # 云端文本补全服务配置（deepseek_v4，OpenAI 兼容 chat-completions 端点）
    # 注意：该端点要求 messages 体；生成节点必须调 CloudCompletionService.chat(messages=...)，
    # 不能用 .complete(prompt=...)（后者发 prompt 字段会被端点拒绝 "error parsing the body"）。
    cloud_completion_url: str = "http://128.23.74.3:9091/AIAPLLM/chat/max/v1/chat/completions"
    cloud_completion_model: str = "deepseek_v4"
    cloud_completion_timeout: int = 60

    # GLM 服务配置
    completion_backend: str = "default"  # default, glm, mock
    glm_api_key: Optional[str] = None
    glm_api_secret: Optional[str] = None
    glm_auth_token: Optional[str] = None
    glm_completion_url: Optional[str] = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
    glm_completion_model: str = "glm-4-7"
    glm_completion_timeout: int = 60

    # MinerU文档解析服务配置
    mineru_url: Optional[str] = None
    mineru_auth_token: Optional[str] = None
    mineru_timeout: int = 300

    # Milvus知识库查询服务配置
    milvus_knowledge_url: str = "http://128.23.77.226:6719/cloudoa-ai/ai/file-knowledge/queryKnowledge"
    milvus_knowledge_token: Optional[str] = None
    milvus_knowledge_timeout: int = 60

    # 测试配置
    test_data_path: str = "./tests/data"
    test_timeout: int = 30

    # ========== New Configuration Options (T017) ==========

    # Elasticsearch 配置
    elasticsearch_hosts: List[str] = ["http://localhost:9200"]
    elasticsearch_index_name: str = "smart_kb_docs"
    elasticsearch_hnsw_m: int = 32
    elasticsearch_hnsw_ef_construction: int = 300
    elasticsearch_hnsw_ef_search: int = 100

    # Structlog 配置
    log_level: str = "INFO"
    log_format: str = "json"  # json or text

    # Prometheus 配置
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # 查询重写配置
    rewrite_rules_path: str = "./config/rewrite_rules.yaml"
    rewrite_enabled: bool = True
    rewrite_llm_expansion_enabled: bool = False

    # 文件监控配置
    watcher_enabled: bool = False
    watcher_scan_interval_seconds: int = 300
    watcher_paths: List[str] = []

    # 搜索默认配置
    search_default_mode: str = "dual"  # sirchmunk, vector, dual
    search_default_top_k: int = 10
    search_max_top_k: int = 100
    search_rerank_enabled: bool = True

    # ========== Milvus (本地 GPU) ==========
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_db: str = "default"
    milvus_collection_prefix: str = "kb_"
    milvus_metric: str = "COSINE"
    milvus_top_k_per_kb: int = 10
    milvus_ef: int = 128
    milvus_hnsw_m: int = 16
    milvus_ef_construction: int = 200

    # ========== Redis (会话 + 缓存) ==========
    redis_url: str = "redis://localhost:6379/0"

    # ========== 入库（模型可插拔） ==========
    parser_backend: str = "mineru"        # mineru | local
    chunker_backend: str = "fixed"        # fixed | recursive
    embedding_backend: str = "cloud"      # cloud (bge-m3)
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_tolerance: int = 50

    # ========== 生成 / 重写 / 稳定性 ==========
    gen_temperature: float = 0.0
    gen_top_p: float = 1.0
    gen_max_tokens: int = 1024
    gen_top_n_context: int = 5
    gen_context_char_per_seg: int = 800
    gen_context_total_chars: int = 4000
    gen_history_turns: int = 3
    rewrite_temperature: float = 0.0
    session_history_messages: int = 6
    session_ttl: int = 86400
    stability_cache_enabled: bool = True
    stability_cache_ttl: int = 604800
    stability_semantic_threshold: float = 0.98

    # ========== KB 分类工具链 (kbmap) ==========
    kbmap_merged_root: str = ""              # 原始 merged/ 目录绝对路径
    kbmap_embed_batch_size: int = 32         # CloudEmbeddingService 批大小
    kbmap_chunk_preview_chars: int = 500     # 嵌入时每文件取首 chunk 字数
    kbmap_outlier_margin: float = 0.05       # 异常文件判定阈值
    kbmap_cohesion_min: float = 0.5          # 类内凝聚度下限
    kbmap_separation_max: float = 0.85       # 类间分离度上限（超过考虑合并）

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# 全局配置实例
settings = Settings()
