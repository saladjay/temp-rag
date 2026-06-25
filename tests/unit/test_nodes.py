"""图节点单元测试：rewrite/retrieve/rerank/generate。

每个节点可独立注入 fake 依赖，不触碰真实后端。
generate_node 为 async；asyncio_mode=auto 下无需显式标记即可 await，
但保留 @pytest.mark.asyncio 以语义清晰。
"""
import pytest

from app.graph.state import ChatState
from app.graph import nodes


# ---------- rewrite_node ----------

def test_rewrite_passthrough_when_no_history():
    state = ChatState(question="怎么退货", history=[])
    out = nodes.rewrite_node(state, llm=None)
    assert out["rewritten_query"] == "怎么退货"


def test_rewrite_uses_llm_when_history_present():
    class FakeLLM:
        def chat(self, messages, **kw):
            assert kw.get("temperature") == 0.0
            return {"text": "iPhone 15 怎么退货"}

    state = ChatState(
        question="那它怎么退货",
        history=[{"role": "user", "content": "iPhone 15 便宜吗"}],
    )
    out = nodes.rewrite_node(state, llm=FakeLLM())
    assert out["rewritten_query"] == "iPhone 15 怎么退货"


def test_rewrite_falls_back_on_llm_error():
    class BoomLLM:
        def chat(self, messages, **kw):
            raise RuntimeError("down")

    state = ChatState(question="原问题", history=[{"role": "user", "content": "x"}])
    out = nodes.rewrite_node(state, llm=BoomLLM())
    assert out["rewritten_query"] == "原问题"


# ---------- retrieve_node ----------

def test_retrieve_writes_back_retrieved_list():
    class FakeEmbedder:
        def embed(self, texts):
            # 返回类 numpy：需支持 [0].tolist()
            class _V:
                def __init__(self, vals):
                    self._v = vals

                def tolist(self):
                    return self._v

            return [_V([0.1, 0.2, 0.3])]

    class FakeHit:
        def __init__(self, doc_id, segment_id, doc_name, text, score, source):
            self.doc_id = doc_id
            self.segment_id = segment_id
            self.doc_name = doc_name
            self.text = text
            self.score = score
            self.source = source

    class FakeStore:
        def search(self, vec, kbs, top_k, ef):
            assert vec == [0.1, 0.2, 0.3]
            assert kbs == ["kb_a"]
            return [
                FakeHit("d1", "d1#0000", "n1", "片段A", 0.9, "kb_a"),
                FakeHit("d2", "d2#0001", "n2", "片段B", 0.8, "kb_a"),
            ]

    state = ChatState(rewritten_query="怎么退货", kb_names=["kb_a"])
    out = nodes.retrieve_node(state, embedder=FakeEmbedder(), store=FakeStore())
    assert [r["doc_id"] for r in out["retrieved"]] == ["d1", "d2"]
    assert out["retrieved"][0]["text"] == "片段A"
    assert out["retrieved"][0]["source"] == "kb_a"


# ---------- rerank_node ----------

def test_rerank_tiebreak_doc_segment_asc():
    # 重排器给两个同分，验证二级排序
    class FakeReranker:
        def rerank(self, query, docs, top_k=None):
            return [{"index": i, "score": s} for i, s in [(0, 0.9), (1, 0.9), (2, 0.5)]]

    retrieved = [
        {"doc_id": "d2", "segment_id": "d2#0001", "doc_name": "n2", "text": "b", "source": "kb_a"},
        {"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1", "text": "a", "source": "kb_a"},
        {"doc_id": "d3", "segment_id": "d3#0002", "doc_name": "n3", "text": "c", "source": "kb_a"},
    ]
    state = ChatState(rewritten_query="q", retrieved=retrieved)
    out = nodes.rerank_node(state, reranker=FakeReranker(), top_n=3)
    ids = [(s["doc_id"], s["segment_id"]) for s in out["sources"]]
    assert ids == [("d1", "d1#0000"), ("d2", "d2#0001"), ("d3", "d3#0002")]


def test_rerank_empty_retrieved_returns_empty():
    state = ChatState(rewritten_query="q", retrieved=[])
    out = nodes.rerank_node(state, reranker=None, top_n=5)
    assert out["sources"] == []


def test_rerank_falls_back_on_reranker_error_keeps_deterministic_order():
    class BoomReranker:
        def rerank(self, query, docs, top_k=None):
            raise RuntimeError("rerank down")

    retrieved = [
        {"doc_id": "d2", "segment_id": "d2#0001", "doc_name": "n2",
         "text": "b", "score": 0.5, "source": "kb_a"},
        {"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1",
         "text": "a", "score": 0.9, "source": "kb_a"},
    ]
    state = ChatState(rewritten_query="q", retrieved=retrieved)
    out = nodes.rerank_node(state, reranker=BoomReranker(), top_n=5)
    # 降级按检索原分排序：score DESC → doc_id ASC → segment_id ASC
    assert [s["doc_id"] for s in out["sources"]] == ["d1", "d2"]


# ---------- generate_node (async) ----------

class _FakeCache:
    """内存 dict 缓存，模拟 DeterministicCache 的 async 接口。"""

    def __init__(self, store=None):
        self._store = store or {}
        self.last_key = None
        self.set_calls = []

    def make_key(self, q, sources_seq, model):
        return f"ans:{q}|{sources_seq}|{model}"

    async def get(self, key):
        self.last_key = key
        return self._store.get(key)

    async def set(self, key, answer):
        self.set_calls.append((key, answer))
        self._store[key] = answer


@pytest.mark.asyncio
async def test_generate_cache_hit_short_circuits_llm():
    cache = _FakeCache()
    # 用 cache 自己的 make_key 预置命中值，避免键格式不一致
    sources = [{"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1",
                "text": "片段", "score": 0.9, "source": "kb_a"}]
    from app.config import settings
    hit_key = cache.make_key(
        "q", [(s["doc_id"], s["segment_id"]) for s in sources],
        settings.cloud_completion_model)
    cache._store[hit_key] = "命中答案"
    llm_called = []

    class SpyLLM:
        def chat(self, messages, **kw):
            llm_called.append(messages)
            return {"text": "不应该被调用"}

    state = ChatState(rewritten_query="q", sources=sources)
    out = await nodes.generate_node(state, llm=SpyLLM(), cache=cache)

    assert out["cache_hit"] is True
    assert out["answer"] == "命中答案"
    assert llm_called == []  # 命中即未触达 LLM


@pytest.mark.asyncio
async def test_generate_miss_calls_llm_and_writes_cache():
    cache = _FakeCache(store={})

    class FakeLLM:
        def chat(self, messages, **kw):
            assert kw.get("temperature") == 0.0
            return {"text": "生成答案"}

    sources = [{"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1",
                "text": "片段内容", "score": 0.9, "source": "kb_a"}]
    state = ChatState(rewritten_query="q", sources=sources)
    out = await nodes.generate_node(state, llm=FakeLLM(), cache=cache)

    assert out["cache_hit"] is False
    assert out["answer"] == "生成答案"
    assert "片段内容" in out["context_text"]
    assert cache.set_calls == [(cache.last_key, "生成答案")]


@pytest.mark.asyncio
async def test_generate_llm_error_returns_error_field():
    cache = _FakeCache(store={})

    class BoomLLM:
        def chat(self, messages, **kw):
            raise RuntimeError("llm down")

    state = ChatState(rewritten_query="q", sources=[
        {"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1",
         "text": "x", "score": 0.9, "source": "kb_a"}])
    out = await nodes.generate_node(state, llm=BoomLLM(), cache=cache)

    assert out["cache_hit"] is False
    assert out["answer"] == ""
    assert "llm down" in out["error"]
    assert cache.set_calls == []  # 失败不写缓存
