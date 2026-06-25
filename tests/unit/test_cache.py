import pytest
import fakeredis.aioredis
from app.store.cache import DeterministicCache


def test_key_deterministic_for_same_inputs():
    c = DeterministicCache()
    sources = [("d1", "d1#0000"), ("d2", "d2#0001")]
    k1 = c.make_key("怎么退货", sources, "Qwen3-32B")
    k2 = c.make_key("怎么退货", sources, "Qwen3-32B")
    assert k1 == k2


def test_key_changes_when_sources_order_changes():
    c = DeterministicCache()
    a = c.make_key("q", [("d1", "s1"), ("d2", "s2")], "m")
    b = c.make_key("q", [("d2", "s2"), ("d1", "s1")], "m")
    assert a != b


def test_key_changes_when_model_changes():
    c = DeterministicCache()
    s = [("d1", "s1")]
    assert c.make_key("q", s, "m1") != c.make_key("q", s, "m2")


@pytest.mark.asyncio
async def test_set_and_get():
    c = DeterministicCache(client=fakeredis.aioredis.FakeRedis())
    k = c.make_key("q", [("d1", "s1")], "m")
    assert await c.get(k) is None
    await c.set(k, "答案是A")
    assert await c.get(k) == "答案是A"


@pytest.mark.asyncio
async def test_get_returns_none_when_disabled(monkeypatch):
    """缓存关闭时 get 应返回 None，set 应空操作。"""
    from app.config import settings
    monkeypatch.setattr(settings, "stability_cache_enabled", False)
    c = DeterministicCache(client=fakeredis.aioredis.FakeRedis())
    k = c.make_key("q", [("d1", "s1")], "m")
    # 缓存关闭：get 返回 None
    assert await c.get(k) is None
    # 缓存关闭：set 空操作，之后再开启也不会有值
    await c.set(k, "答案是A")
    monkeypatch.setattr(settings, "stability_cache_enabled", True)
    assert await c.get(k) is None
