"""编译 LangGraph 状态机。

线性图：START→load_history→rewrite→retrieve→rerank→generate→save_history→END。
session_store 可注入；节点函数引用 app.graph.nodes，测试可通过 monkeypatch 替换。
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.state import ChatState
from app.graph import nodes


def build_graph(session_store=None):
    """编译并返回 LangGraph 状态机。

    :param session_store: 可选的会话历史存储实例（load_history/save_history 注入）。
    :return: 编译后的 CompiledGraph，支持 ainvoke。
    """
    g = StateGraph(ChatState)
    # 使用 async 闭包绑定 session_store；langgraph 依据函数是否 coroutine 自动调度。
    # 其余节点直接引用 nodes.*（便于测试 monkeypatch）。
    async def _load(s):
        return await nodes.load_history_node(s, session_store)

    async def _save(s):
        return await nodes.save_history_node(s, session_store)

    g.add_node("load_history", _load)
    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("rerank", nodes.rerank_node)
    g.add_node("generate", nodes.generate_node)
    g.add_node("save_history", _save)

    g.add_edge(START, "load_history")
    g.add_edge("load_history", "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "save_history")
    g.add_edge("save_history", END)
    return g.compile()
