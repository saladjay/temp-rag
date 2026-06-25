"""Redis 会话历史存储（有界窗口）。"""
from __future__ import annotations
import json
from typing import Optional

from app.config import settings
from app.utils.log import get_logger

logger = get_logger(__name__)


class SessionStore:
    def __init__(self, client=None):
        self._client = client

    async def _r(self):
        if self._client is not None:
            return self._client
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(settings.redis_url)
        return self._client

    async def append(self, session_id: str, role: str, content: str) -> None:
        try:
            r = await self._r()
            key = f"sess:{session_id}"
            await r.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            await r.ltrim(key, -settings.session_history_messages, -1)
            await r.expire(key, settings.session_ttl)
        except Exception as e:
            logger.warning("session_append_failed", error=str(e))

    async def load(self, session_id: str) -> list[dict]:
        try:
            r = await self._r()
            raw = await r.lrange(f"sess:{session_id}", 0, -1)
            return [json.loads(x) for x in raw]
        except Exception as e:
            logger.warning("session_load_failed", error=str(e))
            return []  # 降级：当作首轮

    async def clear(self, session_id: str) -> None:
        r = await self._r()
        await r.delete(f"sess:{session_id}")
