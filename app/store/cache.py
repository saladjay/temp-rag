"""确定性缓存：key=hash(query+sources 序列+模型)，命中即逐字一致。"""
from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

from app.config import settings
from app.utils.log import get_logger

logger = get_logger(__name__)


class DeterministicCache:
    """确定性缓存封装。

    - make_key：对 (rewritten_query, sources_seq, model) 做确定性哈希，
      相同输入 => 相同 key；sources 顺序或 model 变化 => 不同 key。
    - get/set：遵循 settings.stability_cache_enabled 开关；
      Redis 异常或缓存关闭时静默降级（get 返回 None，set 空操作），永不抛错。
    """

    def __init__(self, client=None):
        # 允许外部注入 redis 客户端（如 fakeredis），None 时惰性创建
        self._client = client

    async def _r(self):
        """惰性获取 redis 客户端，未注入时按配置连接真实 Redis。"""
        if self._client is not None:
            return self._client
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(settings.redis_url)
        return self._client

    def make_key(
        self,
        rewritten_query: str,
        sources_seq: Sequence[tuple],
        model: str,
    ) -> str:
        """根据查询、来源序列、模型名生成确定性缓存键。

        使用 json.dumps(sort_keys=True) 保证字段顺序固定，
        再对 utf-8 字节做 sha256，确保纯确定性（无时间/随机因子）。
        """
        payload = json.dumps(
            {
                "q": rewritten_query,
                "sources": [list(s) for s in sources_seq],
                "model": model,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"ans:{h}"

    async def get(self, key: str) -> Optional[str]:
        """读取缓存值。缓存关闭或 Redis 异常时返回 None。"""
        if not settings.stability_cache_enabled:
            return None
        try:
            r = await self._r()
            v = await r.get(key)
            return v.decode("utf-8") if v else None
        except Exception as e:
            logger.warning("cache_get_failed", error=str(e))
            return None  # 降级：跳过缓存

    async def set(self, key: str, answer: str) -> None:
        """写入缓存值。缓存关闭或 Redis 异常时空操作。"""
        if not settings.stability_cache_enabled:
            return
        try:
            r = await self._r()
            await r.set(key, answer, ex=settings.stability_cache_ttl)
        except Exception as e:
            logger.warning("cache_set_failed", error=str(e))
