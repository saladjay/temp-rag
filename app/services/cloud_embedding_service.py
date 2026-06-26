"""云端嵌入服务模块 - 调用云端大模型进行向量嵌入"""
from typing import List, Optional
import numpy as np
import httpx
from app.config import settings


class CloudEmbeddingService:
    """云端嵌入服务类 - 调用远程嵌入API"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
        auth_token: Optional[str] = None
    ):
        """初始化云端嵌入服务

        Args:
            api_url: 嵌入API地址
            model_name: 模型名称
            timeout: 请求超时时间(秒)
            auth_token: 认证令牌
        """
        self.api_url = api_url or settings.cloud_embedding_url
        self.model_name = model_name or settings.cloud_embedding_model
        self.timeout = timeout
        self.auth_token = auth_token or settings.cloud_auth_token

        if not self.api_url:
            raise ValueError("cloud_embedding_url is not configured")

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Basic {self.auth_token}"

        self._client = httpx.Client(timeout=self.timeout, headers=headers, trust_env=False)
        self.last_usage = None   # 最近一次调用的 token 用量

    def encode(self, texts: str | List[str]) -> np.ndarray:
        """将文本编码为向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量数组，形状为 (n, dimension)

        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 响应格式错误
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self._post_embedding_request(texts)
        self.last_usage = response.get("usage")
        return self._parse_embedding_response(response)

    def _post_embedding_request(self, texts: List[str]) -> dict:
        """发送嵌入请求到云端API

        Args:
            texts: 文本列表

        Returns:
            API响应JSON数据
        """
        payload = {
            "input": texts,
            "model": self.model_name
        }

        # 云端可能间歇 502：退避重试（1/2/4s × 4），大批量入库必备
        import time
        max_retries, retry_delay, last_err = 3, 1.0, None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.post(self.api_url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
        raise last_err

    def _parse_embedding_response(self, response: dict) -> np.ndarray:
        """解析API响应，提取嵌入向量

        Args:
            response: API响应JSON数据

        Returns:
            向量数组

        Raises:
            ValueError: 响应格式错误
        """
        # 支持多种响应格式
        if "data" in response:
            # OpenAI 格式: {"data": [{"embedding": [...], ...}, ...]}
            embeddings = [item["embedding"] for item in response["data"]]
        elif "embeddings" in response:
            # 简单格式: {"embeddings": [[...], [...]]}
            embeddings = response["embeddings"]
        elif "embedding" in response:
            # 单一嵌入格式: {"embedding": [...]}
            emb = response["embedding"]
            embeddings = [emb] if isinstance(emb[0], (int, float)) else emb
        else:
            raise ValueError(f"Unexpected response format: {response}")

        return np.array(embeddings, dtype=np.float32)

    def encode_with_retry(
        self,
        texts: str | List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> np.ndarray:
        """带重试机制的文本编码

        Args:
            texts: 单个文本或文本列表
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)

        Returns:
            向量数组
        """
        import time

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.encode(texts)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # 指数退避
                continue
            except Exception as e:
                raise e

        raise last_error if last_error else RuntimeError("Failed to encode texts")

    def get_dimension(self) -> int:
        """获取向量维度

        Returns:
            向量维度
        """
        # 通过发送一个测试请求获取维度
        test_embedding = self.encode("test")
        return test_embedding.shape[1]

    def close(self) -> None:
        """关闭HTTP客户端"""
        self._client.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.close()


class MockCloudEmbeddingService(CloudEmbeddingService):
    """Mock云端嵌入服务，用于测试"""

    def __init__(self, dimension: int = 1024, fixed_seed: int = 42):
        """初始化Mock服务

        Args:
            dimension: 向量维度
            fixed_seed: 随机种子
        """
        self._dimension = dimension
        self._rng = np.random.RandomState(fixed_seed)
        # 调用父类但不初始化HTTP客户端
        object.__setattr__(self, 'api_url', 'mock://')
        object.__setattr__(self, 'model_name', 'mock-model')
        object.__setattr__(self, 'timeout', 30)

    def encode(self, texts: str | List[str]) -> np.ndarray:
        """生成确定性向量"""
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimension).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)

        return np.array(vectors)

    def encode_with_retry(self, texts: str | List[str], max_retries: int = 3, retry_delay: float = 1.0) -> np.ndarray:
        """Mock重试机制，直接返回结果"""
        return self.encode(texts)

    def get_dimension(self) -> int:
        """返回预设维度"""
        return self._dimension

    def close(self) -> None:
        """Mock关闭方法"""
        pass