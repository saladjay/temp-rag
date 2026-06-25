"""Task 8：组装 LangGraph 状态机的端到端连通测试。"""
import pytest
from app.graph.builder import build_graph


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_with_stubs(monkeypatch):
    # 用桩件替换外部依赖，验证图连通
    from app.graph import nodes

    def fake_rewrite(state, llm=None):
        return {"rewritten_query": state["question"]}

    def fake_retrieve(state, embedder=None, store=None):
        return {"retrieved": [{"doc_id": "d1", "segment_id": "d1#0000",
                               "doc_name": "n1", "text": "答案A", "score": 0.9, "source": "kb_a"}]}

    def fake_rerank(state, reranker=None, top_n=None):
        return {"sources": state["retrieved"]}

    def fake_generate(state, llm=None, cache=None, on_token=None):
        return {"answer": "答复", "cache_hit": False, "context_text": "", "cache_key": "k"}

    monkeypatch.setattr(nodes, "rewrite_node", fake_rewrite)
    monkeypatch.setattr(nodes, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(nodes, "rerank_node", fake_rerank)
    monkeypatch.setattr(nodes, "generate_node", fake_generate)

    g = build_graph(session_store=type("S", (), {
        "load": lambda self, sid: [], "append": lambda self, *a: None})())
    result = await g.ainvoke({"session_id": "s1", "question": "你好", "history": []})
    assert result["answer"] == "答复"
    assert result["sources"][0]["doc_id"] == "d1"
