import pytest
import fakeredis.aioredis
from app.store.session_store import SessionStore


@pytest.mark.asyncio
async def test_append_and_load_bounded_window(monkeypatch):
    monkeypatch.setattr("app.config.settings.session_history_messages", 4)
    s = SessionStore(client=fakeredis.aioredis.FakeRedis())
    for i in range(6):
        await s.append("s1", "user", f"q{i}")
    msgs = await s.load("s1")
    assert [m["content"] for m in msgs] == ["q2", "q3", "q4", "q5"]


@pytest.mark.asyncio
async def test_first_turn_empty():
    s = SessionStore(client=fakeredis.aioredis.FakeRedis())
    assert await s.load("new") == []


@pytest.mark.asyncio
async def test_clear():
    s = SessionStore(client=fakeredis.aioredis.FakeRedis())
    await s.append("s2", "user", "x")
    await s.clear("s2")
    assert await s.load("s2") == []
