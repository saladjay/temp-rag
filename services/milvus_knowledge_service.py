"""Milvus向量化知识库查询服务模块

该模块提供对Milvus向量化知识库查询API的封装访问。
支持向量检索、全文检索、混合检索三种检索模式。
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.utils.log import get_logger

logger = get_logger(__name__)


class SearchType(IntEnum):
    """检索类型枚举

    0: 向量检索 - 基于向量相似度匹配
    1: 全文检索 - 基于关键词文本搜索
    2: 混合检索 - 结合向量检索和全文检索
    """

    VECTOR = 0
    FULL_TEXT = 1
    HYBRID = 2


class FileType(str):
    """文件类型常量"""

    RECEIVE = "PublicDocReceive"  # 收文
    DISPATCH = "PublicDocDispatch"  # 发文


class DocMetadata(BaseModel):
    """文档元数据"""

    docid: Optional[str] = Field(None, description="文档ID-OA附件ID")
    doctime: Optional[str] = Field(None, description="文档时间-OA附件更新时间/创建时间")
    doctype: Optional[str] = Field(None, description="文档类型-公文管理-->发文/收文")

    model_config = {"extra": "allow"}


class SegmentMetadata(BaseModel):
    """分段元数据信息"""

    score: Optional[float] = Field(None, description="匹配分数")
    position: Optional[int] = Field(None, description="位置")
    source: Optional[str] = Field(None, alias="_source", description="数据来源-knowledge")
    dataset_id: Optional[str] = Field(None, description="数据集ID-向量库ID")
    dataset_name: Optional[str] = Field(None, description="数据集名称-向量库名")
    document_id: Optional[str] = Field(None, description="文档ID-OA附件ID")
    document_name: Optional[str] = Field(None, description="文档名称-OA附件名称")
    data_source_type: Optional[str] = Field(None, description="数据来源类型-upload_file")
    segment_id: Optional[str] = Field(None, description="分段ID-段落ID")
    retriever_from: Optional[str] = Field(None, description="检索来源-workflow")
    segment_hit_count: Optional[int] = Field(None, description="分段命中次数")
    segment_word_count: Optional[int] = Field(None, description="分段字数")
    segment_position: Optional[int] = Field(None, description="分段位置")
    segment_index_node_hash: Optional[str] = Field(None, description="分段索引节点哈希值")
    doc_metadata: Optional[DocMetadata] = Field(None, description="文档元数据")

    model_config = {"extra": "allow", "populate_by_name": True}


class KnowledgeResult(BaseModel):
    """知识库查询结果项"""

    metadata: SegmentMetadata = Field(..., description="元数据信息")
    title: str = Field(..., description="文件名")
    content: str = Field(..., description="文档内容-段落")

    model_config = {"extra": "allow"}


class KnowledgeQueryRequest(BaseModel):
    """知识库查询请求模型"""

    query: str = Field(..., description="关键词1", min_length=1)
    comp_id: str = Field(..., alias="compId", description="公司集团唯一编码，例如：N000131")
    file_type: str = Field(..., alias="fileType", description="文件类型")
    doc_date: Optional[str] = Field("", alias="docDate", description="文件日期")
    keyword: Optional[str] = Field("", description="关键词2")
    top_k: Optional[int] = Field(10, alias="topk", ge=1, le=100, description="Milvus查询top数量，默认10")
    score_min: Optional[float] = Field(None, alias="scoreMin", ge=0, le=1, description="Score阈值")
    search_type: SearchType = Field(..., alias="searchType", description="检索类型: 0-向量检索、1-全文检索、2-混合检索")

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: str) -> str:
        """验证文件类型"""
        valid_types = [FileType.RECEIVE, FileType.DISPATCH]
        if v not in valid_types:
            raise ValueError(f"file_type must be one of {valid_types}, got: {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: Optional[int]) -> int:
        """验证top_k参数"""
        if v is None:
            return 10
        return v

    model_config = {"populate_by_name": True}


class KnowledgeQueryResponse(BaseModel):
    """知识库查询响应模型"""

    code: int = Field(..., description="状态码，200为成功，失败为500")
    msg: str = Field(..., description="提示信息，成功为Success，失败为错误信息")
    result: list[KnowledgeResult] = Field(default_factory=list, description="查询结果列表")

    @property
    def is_success(self) -> bool:
        """是否查询成功"""
        return self.code == 200

    @property
    def error_message(self) -> str:
        """错误信息"""
        if self.is_success:
            return ""
        return self.msg


class MilvusKnowledgeClient:
    """Milvus知识库查询客户端

    提供对Milvus向量化知识库查询API的封装访问。

    使用示例:
        >>> client = MilvusKnowledgeClient()
        >>> request = KnowledgeQueryRequest(
        ...     query="东方思维",
        ...     comp_id="N000131",
        ...     file_type=FileType.DISPATCH,
        ...     search_type=SearchType.FULL_TEXT,
        ...     top_k=2
        ... )
        >>> response = client.query_knowledge(request)
        >>> if response.is_success:
        ...     for item in response.result:
        ...         print(f"标题: {item.title}")
        ...         print(f"内容: {item.content[:100]}...")
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 60,
    ):
        """初始化Milvus知识库客户端

        Args:
            api_url: API地址，默认从配置读取
            auth_token: 认证令牌，默认从配置读取
            timeout: 请求超时时间(秒)
        """
        self.api_url = api_url or getattr(settings, "milvus_knowledge_url", "http://128.23.77.226:6719/cloudoa-ai/ai/file-knowledge/queryKnowledge")
        self.auth_token = auth_token or getattr(settings, "milvus_knowledge_token", None)
        self.timeout = timeout

        if not self.api_url:
            raise ValueError("milvus_knowledge_url is not configured")

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["xtoken"] = self.auth_token

        self._client = httpx.Client(timeout=self.timeout, headers=headers)

        logger.info(
            "MilvusKnowledgeClient initialized",
            api_url=self.api_url,
            has_auth_token=bool(self.auth_token),
        )

    def query_knowledge(self, request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        """查询知识库

        Args:
            request: 查询请求参数

        Returns:
            查询响应结果

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 响应解析失败
        """
        try:
            payload = request.model_dump(by_alias=True, exclude_none=True)

            logger.info(
                "Sending knowledge query request",
                query=request.query,
                comp_id=request.comp_id,
                file_type=request.file_type,
                search_type=request.search_type.value,
                top_k=request.top_k,
            )

            response = self._client.post(self.api_url, json=payload)
            response.raise_for_status()

            result_data = response.json()
            parsed_response = KnowledgeQueryResponse.model_validate(result_data)

            logger.info(
                "Knowledge query completed",
                success=parsed_response.is_success,
                result_count=len(parsed_response.result),
            )

            return parsed_response

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error occurred during knowledge query",
                status_code=e.response.status_code,
                error=str(e),
            )
            # 尝试解析错误响应
            try:
                error_data = e.response.json()
                return KnowledgeQueryResponse(
                    code=500,
                    msg=f"HTTP {e.response.status_code}: {error_data.get('msg', str(e))}",
                    result=[],
                )
            except Exception:
                return KnowledgeQueryResponse(
                    code=500,
                    msg=f"HTTP {e.response.status_code}: {str(e)}",
                    result=[],
                )

        except Exception as e:
            logger.exception("Unexpected error during knowledge query")
            return KnowledgeQueryResponse(code=500, msg=f"查询失败: {str(e)}", result=[])

    def query_knowledge_simple(
        self,
        query: str,
        comp_id: str,
        file_type: str,
        search_type: SearchType,
        keyword: str = "",
        doc_date: str = "",
        top_k: int = 10,
        score_min: Optional[float] = None,
    ) -> KnowledgeQueryResponse:
        """简化版知识库查询方法

        Args:
            query: 查询关键词
            comp_id: 公司ID
            file_type: 文件类型
            search_type: 检索类型
            keyword: 可选的第二关键词
            doc_date: 可选的文件日期
            top_k: 返回结果数量
            score_min: 可选的分数阈值

        Returns:
            查询响应结果
        """
        request = KnowledgeQueryRequest(
            query=query,
            comp_id=comp_id,
            file_type=file_type,
            search_type=search_type,
            keyword=keyword,
            doc_date=doc_date,
            top_k=top_k,
            score_min=score_min,
        )
        return self.query_knowledge(request)

    def search_documents(
        self,
        query: str,
        comp_id: str,
        file_type: str,
        top_k: int = 10,
        mode: str = "hybrid",
        keyword: str = "",
        doc_date: str = "",
        score_min: Optional[float] = None,
    ) -> tuple[list[KnowledgeResult], str]:
        """搜索文档（兼容接口）

        Args:
            query: 查询关键词
            comp_id: 公司ID
            file_type: 文件类型
            top_k: 返回结果数量
            mode: 搜索模式 (vector/full_text/hybrid)
            keyword: 可选的第二关键词
            doc_date: 可选的文件日期
            score_min: 可选的分数阈值

        Returns:
            (结果列表, 错误信息)
        """
        mode_map = {
            "vector": SearchType.VECTOR,
            "full_text": SearchType.FULL_TEXT,
            "hybrid": SearchType.HYBRID,
        }
        search_type = mode_map.get(mode, SearchType.HYBRID)

        response = self.query_knowledge_simple(
            query=query,
            comp_id=comp_id,
            file_type=file_type,
            search_type=search_type,
            keyword=keyword,
            doc_date=doc_date,
            top_k=top_k,
            score_min=score_min,
        )

        if response.is_success:
            return response.result, ""
        return [], response.error_message

    def close(self) -> None:
        """关闭HTTP客户端"""
        self._client.close()
        logger.info("MilvusKnowledgeClient closed")

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.close()


class MockMilvusKnowledgeClient(MilvusKnowledgeClient):
    """Mock Milvus知识库客户端，用于测试"""

    def __init__(self, fixed_response: Optional[KnowledgeQueryResponse] = None):
        """初始化Mock客户端

        Args:
            fixed_response: 固定的响应结果，用于测试
        """
        object.__setattr__(self, "api_url", "mock://")
        object.__setattr__(self, "auth_token", "mock_token")
        object.__setattr__(self, "timeout", 60)
        object.__setattr__(self, "_fixed_response", fixed_response)

    def query_knowledge(self, request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        """Mock查询方法"""
        logger.info(
            "Mock knowledge query",
            query=request.query,
            comp_id=request.comp_id,
            file_type=request.file_type,
            search_type=request.search_type.value,
        )

        if self._fixed_response:
            return self._fixed_response

        # 返回默认Mock响应
        mock_result = KnowledgeResult(
            metadata=SegmentMetadata(
                score=0.95,
                position=1,
                source="knowledge",
                dataset_id=f"{request.comp_id}_{request.file_type}",
                dataset_name=f"{request.comp_id}_{request.file_type}",
                document_id="mock_doc_id",
                document_name="mock_document.pdf",
                data_source_type="upload_file",
                segment_id="mock_segment_id",
                retriever_from="workflow",
                segment_hit_count=1,
                segment_word_count=100,
                segment_position=0,
                doc_metadata=DocMetadata(
                    docid="mock_doc_id",
                    doctime="2024-01-01",
                    doctype="公文管理-->发文",
                ),
            ),
            title=f"Mock文档: {request.query}",
            content=f"这是关于 '{request.query}' 的Mock内容。这是一个测试响应，用于在没有实际API连接时进行开发测试。",
        )

        return KnowledgeQueryResponse(
            code=200,
            msg="Success",
            result=[mock_result],
        )

    def close(self) -> None:
        """Mock关闭方法"""
        pass


# 便捷函数
def get_milvus_client() -> MilvusKnowledgeClient:
    """获取Milvus知识库客户端实例

    Returns:
        MilvusKnowledgeClient实例
    """
    return MilvusKnowledgeClient()


def query_knowledge(
    query: str,
    comp_id: str,
    file_type: str,
    search_type: SearchType = SearchType.FULL_TEXT,
    keyword: str = "",
    doc_date: str = "",
    top_k: int = 10,
    score_min: Optional[float] = None,
) -> KnowledgeQueryResponse:
    """便捷函数：查询知识库

    Args:
        query: 查询关键词
        comp_id: 公司ID
        file_type: 文件类型
        search_type: 检索类型
        keyword: 可选的第二关键词
        doc_date: 可选的文件日期
        top_k: 返回结果数量
        score_min: 可选的分数阈值

    Returns:
        查询响应结果
    """
    client = get_milvus_client()
    try:
        return client.query_knowledge_simple(
            query=query,
            comp_id=comp_id,
            file_type=file_type,
            search_type=search_type,
            keyword=keyword,
            doc_date=doc_date,
            top_k=top_k,
            score_min=score_min,
        )
    finally:
        client.close()
