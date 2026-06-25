"""验收口径自动化：同一问题跑 N 次，断言检索序列严格一致 + 答案语义稳定。"""
import pytest
from app.graph import nodes
from app.graph.state import ChatState


class StableFakeStore:
    """返回固定命中序列，模拟稳定检索。"""
    def list_kbs(self):
        return ["kb_a"]
    def search(self, vec, kbs, top_k, ef):
        from app.store.milvus_store import MilvusSearchHit
        return [MilvusSearchHit(pk=1, score=0.9, text="答复A", doc_id="d1",
                                doc_name="n1", segment_id="d1#0000", source="kb_a"),
                MilvusSearchHit(pk=2, score=0.5, text="其他", doc_id="d2",
                                doc_name="n2", segment_id="d2#0001", source="kb_a")]


class StableFakeEmbedder:
    def embed(self, texts):
        import numpy as np
        return np.array([[0.1, 0.2, 0.3, 0.4]] * len(texts), dtype="float32")


class IdentityReranker:
    def rerank(self, query, docs, top_k=None):
        return [{"index": i, "score": 1.0 - i * 0.1} for i in range(len(docs))]


def test_retrieval_sequence_strictly_consistent_across_runs():
    state = ChatState(session_id="s", question="q", history=[],
                      rewritten_query="q", kb_names=["kb_a"])
    seqs = []
    for _ in range(10):
        r = nodes.retrieve_node(state, embedder=StableFakeEmbedder(), store=StableFakeStore())
        o = nodes.rerank_node({**state, **r}, reranker=IdentityReranker(), top_n=5)
        seqs.append([(s["doc_id"], s["segment_id"]) for s in o["sources"]])
    # 10 次完全一致
    assert all(s == seqs[0] for s in seqs)
    assert seqs[0] == [("d1", "d1#0000"), ("d2", "d2#0001")]


@pytest.mark.asyncio
async def test_answer_semantic_stability_via_cache():
    # 命中缓存 → 逐字一致（语义稳定的强保证）
    import fakeredis.aioredis
    from app.store.cache import DeterministicCache
    cache = DeterministicCache(client=fakeredis.aioredis.FakeRedis())
    sources = [{"doc_id": "d1", "segment_id": "d1#0000", "doc_name": "n1",
                "text": "t", "score": 0.9, "source": "kb_a"}]
    state = ChatState(session_id="s", question="q", history=[], rewritten_query="q", sources=sources)

    class FixedLLM:
        def __init__(self): self.calls = 0
        def chat(self, messages, **kw):
            self.calls += 1
            return {"text": "固定答复"}

    llm = FixedLLM()
    a1 = (await nodes.generate_node(state, llm=llm, cache=cache))["answer"]
    a2 = (await nodes.generate_node(state, llm=llm, cache=cache))["answer"]
    assert a1 == a2                      # 逐字一致
    assert llm.calls == 1               # 第二次命中缓存未再调 LLM
