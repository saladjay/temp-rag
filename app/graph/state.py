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
