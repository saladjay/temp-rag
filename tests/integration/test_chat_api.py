"""集成测试：/chat SSE 端点（缓存命中路径，不打外部）。"""
import pytest
from app.main import create_app
from app.store.cache import DeterministicCache
from app.graph import nodes


@pytest.mark.asyncio
async def test_chat_returns_cached_answer(monkeypatch):
    # 预置缓存命中：generate_node 走 cache 分支
    async def fake_load(s, store=None):
        return {"history": []}

    async def fake_save(s, store=None):
        return {}

    monkeypatch.setattr(nodes, "load_history_node", fake_load)
    monkeypatch.setattr(nodes, "save_history_node", fake_save)
    monkeypatch.setattr(
        nodes,
        "rewrite_node",
        lambda s, llm=None: {"rewritten_query": s["question"]},
    )
    monkeypatch.setattr(
        nodes,
        "retrieve_node",
        lambda s, embedder=None, store=None: {
            "retrieved": [
                {
                    "doc_id": "d1",
                    "segment_id": "d1#0000",
                    "doc_name": "n1",
                    "text": "t",
                    "score": 0.9,
                    "source": "kb_a",
                }
            ]
        },
    )
    monkeypatch.setattr(
        nodes,
        "rerank_node",
        lambda s, reranker=None, top_n=None: {"sources": s["retrieved"]},
    )

    cache = DeterministicCache(
        client=__import__("fakeredis.aioredis", fromlist=["FakeRedis"]).FakeRedis()
    )

    async def fake_generate(s, llm=None, cache=None, on_token=None):
        return {
            "answer": "缓存答复",
            "cache_hit": True,
            "context_text": "",
            "cache_key": "k",
        }

    monkeypatch.setattr(nodes, "generate_node", fake_generate)

    app = create_app(testing=True)
    from starlette.testclient import TestClient

    with TestClient(app) as tc:
        resp = tc.post("/api/v1/chat", json={"session_id": "s1", "question": "你好"})
    assert resp.status_code == 200
    body = "".join(l for l in resp.text.splitlines() if l.startswith("data:"))
    assert "缓存答复" in body


def test_health_ok():
    """健康检查端点返回 200。"""
    from app.main import create_app
    from starlette.testclient import TestClient

    app = create_app(testing=True)
    with TestClient(app) as tc:
        resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_clear_session(monkeypatch):
    """会话清空端点调用 SessionStore.clear。"""
    from app.main import create_app
    from starlette.testclient import TestClient
    from app.api import routes

    cleared = {}

    class FakeStore:
        async def clear(self, session_id):
            cleared["sid"] = session_id

    monkeypatch.setattr(routes, "SessionStore", lambda: FakeStore())

    app = create_app(testing=True)
    with TestClient(app) as tc:
        resp = tc.post("/api/v1/sessions/abc/clear")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert cleared["sid"] == "abc"
