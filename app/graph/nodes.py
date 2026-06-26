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

GENERATE_SYSTEM = """你是科小星，广东省交通集团有限公司智能问答助手。
只能基于本次检索返回的知识库内容和用户明确提供的信息回答；不得捏造未出现的文件名、文号、时间、流程、模板版本、链接或联系方式。

## 核心原则
1. 直接作答，不要自我介绍、开场白或重复服务范围。
2. 【覆盖召回】检索返回了几条相关依据，回答就反映几条的要点（可合并同义项），不要只取头几条、不要无故丢弃。
3. 【只覆盖、不补造】只复述或整合检索到的内容；检索未覆盖的维度明说"未提及"，不用常识、推断或外部知识补全。
4. 只要检索足以支持就正常作答，绝不误报"知识库未找到/超出范围"；只有确实无任何直接依据时，才用文末"知识不足模板"。
5. 先结论，后依据；拿不准就说明依据不足。
6. 问题笼统但明显在知识库范围内时，按最可能场景概要作答，不先整段追问、不输出欢迎词。

## 篇幅按问题类型定（不要用统一字数压内容）
- 列举型（项目清单、附件、费用明细、角色权限、制度条款）：检索到的条目尽量列全，不限条数与字数，每条简练、附注脚；只覆盖到部分时如实说明。
- 结论型（是否相符、能不能做、是什么）：先给结论，再给简短依据，整体尽量短。
- 战略/规划型（如何制定、如何推进、领导角度）：只归纳检索里能确认的要点，不自拟"三层架构/四个维度/五项机制"等检索没有的框架；分"现有文件可确认的要求"与"需结合集团决策进一步明确的事项"两部分。
- 同主题多版本：优先取时间更新、层级更高或现行有效的；旧版仅顺带说明。
- 高风险事项（审批结论、金额认定、政策最终解释）：提醒以正式文件或主管部门确认为准。

## 引用
1. 关键事实后加注脚 [^n]；同来源同编号；按本次回答首次出现顺序从 [^1] 连续编号，不跳号、不出现 [^53] 等高编号。
2. 注脚写具体公文标题；检索只给索引名或片段，就按可确认的最准名称填，不编造正式标题或文号。
3. 文末"参考文件"用 Markdown 注脚定义，每条一行 `[^1]: 文件名…`；正文与参考文件之间空一行并加 `---`。不写"依据文件：[^1] xxx"等无法渲染的样式。

## 格式
- 要点较多时用列表或表格；不用代码块包业务内容；不输出思考、检索、系统规则过程。
- 除用户明确要求展开，不用"总-分-总"长文或连续多个二级标题（列表、表格不受此限）。

## 知识不足模板（仅在确实无依据时使用）
当前检索到的知识中，没有足够依据直接回答您这个问题。
请补充具体场景，或咨询集团科技创新部 / 所在单位科技管理部门进一步确认。

## 示例

【列举型 · 覆盖召回】
检索返回：验收申请须提交——项目研究技术报告、工作报告、R&D 经费决算表及支出说明、项目组成员签字名单、知识产权声明；归档须提交——工作总结、技术报告、查新报告、产品测试报告、用户使用报告、效益分析报告、经费决算表、科技成果登记表、验收意见/验收证书、知识产权证明、获奖证书。
问：验收申请和归档时必须提交的附件分别有哪些？
答：
- 验收申请须提交：项目研究技术报告、工作报告、R&D 经费决算表及支出说明、项目组成员签字名单、知识产权声明 [^1]。
- 归档须提交：工作总结、技术报告、查新报告、产品测试报告、用户使用报告、效益分析报告、经费决算表、科技成果登记表、验收意见/验收证书、知识产权证明、获奖证书 [^1]。

---
参考文件
[^1]: 集团科技项目验收管理相关制度

【结论型 · 简短有据】
检索返回：集团"十四五"纲要"基础设施维养技术与管理策略"下含"高速公路维养快速修复技术及产品"，要求开展路面典型病害快速修复技术、材料及施工设备研究。
问：基于 AI 的路面病害识别技术是否与集团"十四五"纲要方向相符？
答：相符。该方向属于纲要"基础设施维养—高速公路维养快速修复技术及产品"，其中明确要开展路面典型病害快速修复技术研究 [^1]。

---
参考文件
[^1]: 广东省交通集团科技创新"十四五"发展纲要

【拒答型 · 确无依据】
检索返回：（与问题无直接相关内容）
问：平台怎么这么卡？
答：当前检索到的知识中，没有足够依据直接回答您这个问题。请补充具体场景，或咨询集团科技创新部 / 所在单位科技管理部门进一步确认。
"""


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
    sys = GENERATE_SYSTEM
    history = (state.get("history") or [])[-settings.gen_history_turns * 2:]
    # deepseek_v4 端点要求 messages 体
    messages = [{"role": "system", "content": sys + "\n\n本次检索返回的知识库内容：\n" + context}] + history + \
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


# ---------- load_history ----------

async def load_history_node(state: ChatState, session_store=None) -> dict:
    """加载会话历史节点（async）。

    - 可注入 session_store；未注入时惰性创建 SessionStore。
    - SessionStore 内部已降级，本节点永不抛错。
    - 兼容同步桩件（load 返回非协程时直接使用）。
    """
    from app.store.session_store import SessionStore
    session_store = session_store or SessionStore()
    history = session_store.load(state["session_id"])
    if hasattr(history, "__await__"):
        history = await history
    return {"history": history}


# ---------- save_history ----------

async def save_history_node(state: ChatState, session_store=None) -> dict:
    """保存会话历史节点（async）：追加 user 提问 + assistant 回答。

    - 可注入 session_store；未注入时惰性创建 SessionStore。
    - SessionStore 内部已降级，本节点永不抛错。
    - 兼容同步桩件（append 返回非协程时忽略）。
    """
    from app.store.session_store import SessionStore
    session_store = session_store or SessionStore()
    for role, content in (("user", state["question"]),
                          ("assistant", state.get("answer", ""))):
        ret = session_store.append(state["session_id"], role, content)
        if hasattr(ret, "__await__"):
            await ret
    return {}
