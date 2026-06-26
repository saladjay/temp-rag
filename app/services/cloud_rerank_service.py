"""云端重排序服务模块 - 调用云端重排序API"""
from typing import List, Optional, Tuple, TypedDict
import httpx
from app.config import settings


class RerankResult(TypedDict):
    """重排序结果类型"""
    index: int
    document: str
    score: float


class CloudRerankService:
    """云端重排序服务类 - 调用远程重排序API"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
        auth_token: Optional[str] = None
    ):
        """初始化云端重排序服务

        Args:
            api_url: 重排序API地址
            model_name: 模型名称
            timeout: 请求超时时间(秒)
            auth_token: 认证令牌
        """
        self.api_url = api_url or settings.cloud_rerank_url
        self.model_name = model_name or settings.cloud_rerank_model
        self.timeout = timeout
        self.auth_token = auth_token or settings.cloud_auth_token

        if not self.api_url:
            raise ValueError("cloud_rerank_url is not configured")

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Basic {self.auth_token}"

        self._client = httpx.Client(timeout=self.timeout, headers=headers, trust_env=False)
        self.last_usage = None   # 最近一次调用的 token 用量

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """对文档列表进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果，None表示返回全部

        Returns:
            重排序后的结果列表，包含索引、文档和分数

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 响应格式错误
        """
        response = self._post_rerank_request(query, documents)
        self.last_usage = response.get("usage")
        results = self._parse_rerank_response(response, documents)

        if top_k is not None:
            results = results[:top_k]

        return results

    def _post_rerank_request(self, query: str, documents: List[str]) -> dict:
        """发送重排序请求到云端API

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            API响应JSON数据
        """
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents
        }

        response = self._client.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()

    def _parse_rerank_response(
        self,
        response: dict,
        original_documents: List[str]
    ) -> List[RerankResult]:
        """解析API响应，提取重排序结果

        Args:
            response: API响应JSON数据
            original_documents: 原始文档列表

        Returns:
            重排序结果列表

        Raises:
            ValueError: 响应格式错误
        """
        # 支持多种响应格式
        if "results" in response:
            # Cohere 风格: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            results = [
                {
                    "index": item.get("index", i),
                    "document": original_documents[item.get("index", i)],
                    "score": item.get("relevance_score", item.get("score", 0.0))
                }
                for i, item in enumerate(response["results"])
            ]
        elif "data" in response:
            # 简单格式: {"data": [{"index": 0, "score": 0.95}, ...]}
            results = [
                {
                    "index": item.get("index", i),
                    "document": original_documents[item.get("index", i)],
                    "score": item.get("score", 0.0)
                }
                for i, item in enumerate(response["data"])
            ]
        elif "scores" in response:
            # 分数数组格式: {"scores": [0.95, 0.80, 0.75]}
            scores = response["scores"]
            indexed_scores = [(i, score) for i, score in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            results = [
                {
                    "index": idx,
                    "document": original_documents[idx],
                    "score": score
                }
                for idx, score in indexed_scores
            ]
        else:
            raise ValueError(f"Unexpected response format: {response}")

        return results

    def rerank_with_retry(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[RerankResult]:
        """带重试机制的重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)

        Returns:
            重排序结果列表
        """
        import time

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.rerank(query, documents, top_k)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
            except Exception as e:
                raise e

        raise last_error if last_error else RuntimeError("Failed to rerank documents")

    def rerank_and_extract(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """重排序并分别返回文档和分数

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果

        Returns:
            (重排序后的文档列表, 对应的分数列表)
        """
        results = self.rerank(query, documents, top_k)
        reranked_docs = [r["document"] for r in results]
        scores = [r["score"] for r in results]
        return reranked_docs, scores

    def close(self) -> None:
        """关闭HTTP客户端"""
        self._client.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.close()


class MockCloudRerankService(CloudRerankService):
    """Mock云端重排序服务，用于测试"""

    def __init__(self, fixed_seed: int = 42):
        """初始化Mock服务

        Args:
            fixed_seed: 随机种子
        """
        import numpy as np
        self._rng = np.random.RandomState(fixed_seed)
        object.__setattr__(self, 'api_url', 'mock://')
        object.__setattr__(self, 'model_name', 'mock-rerank-model')
        object.__setattr__(self, 'timeout', 30)

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Mock重排序，基于文本相似度生成分数"""
        results = []
        for i, doc in enumerate(documents):
            # 简单的文本匹配计算分数
            score = 0.0
            query_lower = query.lower()
            doc_lower = doc.lower()

            # 查询词在文档中出现
            if query_lower in doc_lower:
                score = 0.8

            # 添加一些随机性使结果更真实
            score += self._rng.random() * 0.2

            results.append({
                "index": i,
                "document": doc,
                "score": min(score, 1.0)
            })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_retry(self, query: str, documents: List[str], top_k: Optional[int] = None, max_retries: int = 3, retry_delay: float = 1.0) -> List[RerankResult]:
        """Mock重试机制"""
        return self.rerank(query, documents, top_k)

    def close(self) -> None:
        """Mock关闭方法"""
        pass