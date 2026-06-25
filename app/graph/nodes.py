"""LangGraph 节点函数（可独立测试）。

每个节点接受可注入的依赖（llm/embedder/store/reranker/cache），
未注入时惰性创建真实服务实例；生产由图组装层（Task 8）接线。
注意：deepseek_v4 端点要求 messages 体，因此 rewrite/generate 均使用
llm.chat(messages=...) 而非 llm.complete(prompt=...)。
"""
from __future__ import annotations
from typing import Optional, Callable

from app.config import settings
from app.graph.state import ChatState
from app.utils.log import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM = (
    "你是查询重写助手。根据对话历史，把用户的最新提问改写成一个独立、完整、"
    "可脱离上下文检索的问题。只输出改写后的问题，不要解释。"
)


def _default_llm():
    """惰性创建默认补全服务（deepseek_v4，chat-completions 端点）。"""
    from app.services import CloudCompletionService
    return CloudCompletionService()


def _default_embedder():
    """惰性创建默认嵌入器。"""
    from app.ingest.embedder import CloudEmbedder
    return CloudEmbedder()


def _default_reranker():
    """惰性创建默认重排器。"""
    from app.services import CloudRerankService
    return CloudRerankService()


# ---------- rewrite ----------

def rewrite_node(state: ChatState, llm=None) -> dict:
    """查询重写节点。

    - 无历史：直接透传原问题（不触达 LLM）。
    - 有历史：调 LLM（temperature=settings.rewrite_temperature）重写为独立问题。
    - LLM 异常：降级返回原问题，永不抛错。
    """
    question = state["question"]
    history = state.get("history") or []
    if not history:
        return {"rewritten_query": question}
    llm = llm or _default_llm()
    try:
        # deepseek_v4 端点要求 messages 体：system + history + user
        messages = [{"role": "system", "content": REWRITE_SYSTEM}] + history + \
                   [{"role": "user", "content": question}]
        resp = llm.chat(messages, temperature=settings.rewrite_temperature, max_tokens=128)
        return {"rewritten_query": (resp.get("text") or "").strip() or question}
    except Exception as e:
        logger.warning("rewrite_failed_fallback", error=str(e))
        return {"rewritten_query": question}  # 降级


# ---------- retrieve ----------

def retrieve_node(state: ChatState, embedder=None, store=None) -> dict:
    """并行多知识库检索节点：embed → 多 collection 检索 → 写回 retrieved。"""
    from app.store.milvus_store import MilvusStore
    embedder = embedder or _default_embedder()
    store = store or MilvusStore()
    q = state["rewritten_query"]
    vec = embedder.embed([q])[0].tolist()
    kbs = state.get("kb_names") or store.list_kbs()
    hits = store.search(vec, kbs, top_k=settings.milvus_top_k_per_kb, ef=settings.milvus_ef)
    retrieved = [{"doc_id": h.doc_id, "segment_id": h.segment_id, "doc_name": h.doc_name,
                  "text": h.text, "score": h.score, "source": h.source} for h in hits]
    return {"retrieved": retrieved}


# ---------- rerank ----------

def rerank_node(state: ChatState, reranker=None, top_n: Optional[int] = None) -> dict:
    """重排节点：确定排序 score DESC → doc_id ASC → segment_id ASC。

    - 重排器异常：降级用检索原分排序（同样确定性二级键）。
    - 空 retrieved：直接返回空列表（不触达重排器）。
    """
    top_n = top_n or settings.gen_top_n_context
    retrieved = state.get("retrieved") or []
    if not retrieved:
        return {"sources": []}
    reranker = reranker or _default_reranker()
    docs = [r["text"] for r in retrieved]
    try:
        results = reranker.rerank(state["rewritten_query"], docs, top_k=top_n)
    except Exception as e:
        logger.warning("rerank_failed_fallback", error=str(e))
        # 降级：用检索原分排序（同确定二级键）
        ordered = sorted(retrieved, key=lambda r: (-r.get("score", 0), r["doc_id"], r["segment_id"]))
        return {"sources": ordered[:top_n]}
    # 二级排序：score DESC → doc_id+segment_id ASC
    pairs = []
    for r in results:
        idx = r["index"]
        src = retrieved[idx]
        pairs.append((r["score"], src["doc_id"], src["segment_id"], src))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return {"sources": [p[3] for p in pairs[:top_n]]}


# ---------- generate ----------

def build_context_text(sources: list[dict]) -> str:
    """按确定顺序拼接上下文，固定截断（per-seg + total 字符上限）。"""
    parts, total = [], 0
    for s in sources:
        seg = s["text"][:settings.gen_context_char_per_seg]
        if total + len(seg) > settings.gen_context_total_chars:
            seg = seg[: settings.gen_context_total_chars - total]
        parts.append(f"【来源：{s['doc_name']}】{seg}")
        total += len(seg)
        if total >= settings.gen_context_total_chars:
            break
    return "\n\n".join(parts)


async def generate_node(
    state: ChatState,
    llm=None,
    cache=None,
    on_token: Optional[Callable] = None,
) -> dict:
    """生成节点（async）：缓存检查 → 构建确定上下文 → LLM → 写缓存。

    - 命中缓存：cache_hit=True，不触达 LLM。
    - 未命中：构建上下文 → LLM（temperature=settings.gen_temperature）→ 写缓存。
    - LLM 异常：返回 error 字段，不写缓存。
    - 流式回调 on_token 在 Task 9 接入；本任务为非流式。
    """
    from app.store.cache import DeterministicCache
    sources = state.get("sources") or []
    cache = cache or DeterministicCache()
    model = settings.cloud_completion_model
    key = cache.make_key(state["rewritten_query"],
                         [(s["doc_id"], s["segment_id"]) for s in sources], model)
    cached = await cache.get(key)
    if cached is not None:
        return {"answer": cached, "cache_key": key, "cache_hit": True, "context_text": ""}
    # 未命中 → 生成（非流式；token 级流式见 Task 9）
    llm = llm or _default_llm()
    context = build_context_text(sources)
    sys = ("你是客服助手，只能依据下方知识库内容回答；无依据时回答\"知识库中无相关内容\"。"
           "作答简洁、稳定、客观，不编造。")
    history = (state.get("history") or [])[-settings.gen_history_turns * 2:]
    # deepseek_v4 端点要求 messages 体
    messages = [{"role": "system", "content": sys + "\n\n知识库：\n" + context}] + history + \
               [{"role": "user", "content": state["rewritten_query"]}]
    try:
        resp = llm.chat(messages, temperature=settings.gen_temperature,
                        top_p=settings.gen_top_p, max_tokens=settings.gen_max_tokens)
        answer = resp.get("text") or ""
    except Exception as e:
        logger.exception("generate_failed")
        return {"answer": "", "cache_key": key, "cache_hit": False,
                "context_text": context, "error": str(e)}
    await cache.set(key, answer)
    return {"answer": answer, "cache_key": key, "cache_hit": False, "context_text": context}
