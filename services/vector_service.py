"""向量服务模块 - 处理向量嵌入和相似度计算"""
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from app.config import settings

# sentence_transformers 是可选依赖，仅在需要时导入
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass


class VectorService:
    """向量服务类"""

    def __init__(self, model_name: Optional[str] = None):
        """初始化向量服务

        Args:
            model_name: 嵌入模型名称，默认使用配置中的模型
        """
        self.model_name = model_name or settings.embedding_model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """懒加载嵌入模型"""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: str | List[str]) -> np.ndarray:
        """将文本编码为向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量数组，形状为 (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数，范围[0,1]
        """
        # 确保是一维向量
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def batch_similarity(self, query_vec: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """批量计算相似度

        Args:
            query_vec: 查询向量
            doc_vectors: 文档向量数组

        Returns:
            相似度分数数组
        """
        # 确保是一维查询向量
        query_vec = query_vec.flatten()

        # 批量计算余弦相似度
        dot_products = np.dot(doc_vectors, query_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vectors, axis=1)

        similarities = dot_products / (query_norm * doc_norms + 1e-8)
        return similarities

    def get_dimension(self) -> int:
        """获取向量维度"""
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is not installed. Install it with: uv sync --extra vector")
        return self.model.get_sentence_embedding_dimension()


class MockVectorService(VectorService):
    """Mock向量服务，用于测试"""

    def __init__(self, dimension: int = 384, fixed_seed: int = 42):
        """初始化Mock向量服务

        Args:
            dimension: 向量维度
            fixed_seed: 随机种子，保证可重复性
        """
        self._dimension = dimension
        self._rng = np.random.RandomState(fixed_seed)
        self._model = None  # 不加载真实模型

    @property
    def model(self) -> None:
        """Mock模型，返回None"""
        return None

    def encode(self, texts: str | List[str]) -> np.ndarray:
        """生成固定但随机的向量"""
        if isinstance(texts, str):
            texts = [texts]

        # 使用字符串哈希生成确定性向量
        vectors = []
        for text in texts:
            # 使用文本内容的哈希作为种子
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimension)
            # 归一化
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)

        return np.array(vectors)

    def get_dimension(self) -> int:
        """返回预设维度"""
        return self._dimension
