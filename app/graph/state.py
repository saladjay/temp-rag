"""LangGraph 状态定义。"""
from __future__ import annotations
from typing import TypedDict, Optional, Any


class ChatState(TypedDict, total=False):
    session_id: str
    question: str
    kb_names: list[str]
    history: list[dict]
    rewritten_query: str
    retrieved: list[Any]        # list[MilvusSearchHit]
    sources: list[dict]         # [{doc_id,segment_id,doc_name,text,score,source}]
    context_text: str
    answer: str
    cache_key: str
    cache_hit: bool
    error: Optional[str]
    retrieve_trace: Optional[dict]   # 检索耗时+IO: {query, kbs, embed_ms, search_ms, total_ms, n_hits, embed_usage, top3}
    rerank_trace: Optional[dict]     # 重排耗时+IO: {query, n_in, ms, usage, fallback, top}
    generate_trace: Optional[dict]   # 生成耗时+token: {model, gen_ms, usage{prompt/completion/total}, n_sources, context_chars}
