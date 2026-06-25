"""FastAPI 路由：/chat(SSE)、/sessions、/health。"""
from __future__ import annotations
import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest
from app.config import settings
from app.graph.builder import build_graph
from app.store.session_store import SessionStore
from app.utils.log import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix=settings.api_prefix)
# 健康检查挂根路径（GET /health），独立于 api_prefix
health_router = APIRouter()


def _sse(event: str, data) -> dict:
    """构造 SSE 单条消息字典（event + json data）。"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答端点。

    每请求编译一次图（v1 简化）；依次发 rewrite/sources/done 事件，
    异常时发 error 事件并结束（不崩溃服务）。
    """
    # 每请求编译图：节点引用 app.graph.nodes.*，便于测试 monkeypatch
    graph = build_graph(session_store=SessionStore())
    init = {
        "session_id": req.session_id,
        "question": req.question,
        "history": [],
        "kb_names": req.kb_names,
    }

    async def event_gen():
        try:
            # 非流式 invoke；token 级流式为后续扩展（见 brief 说明）
            result = await graph.ainvoke(init)
            sources = result.get("sources") or []
            yield _sse("rewrite", {"rewritten_query": result.get("rewritten_query", "")})
            yield _sse(
                "sources",
                [
                    {
                        "doc_name": s["doc_name"],
                        "score": s["score"],
                        "source": s["source"],
                    }
                    for s in sources
                ],
            )
            yield _sse(
                "done",
                {
                    "answer": result.get("answer", ""),
                    "sources": sources,
                    "cache_hit": result.get("cache_hit", False),
                },
            )
        except Exception as e:
            logger.exception("chat_failed")
            yield _sse("error", {"msg": str(e)})

    return EventSourceResponse(event_gen())


@router.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """清空指定会话历史。"""
    await SessionStore().clear(session_id)
    return {"ok": True}


@health_router.get("/health")
async def health():
    """健康检查（挂根路径）。"""
    return {"status": "ok"}
