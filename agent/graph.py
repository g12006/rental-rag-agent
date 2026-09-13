"""
LangGraph 主图：租房平台智能客服

编排结构（对应 LangGraph 课堂代码的多个知识点）：

    START
      └─> sync_profile           跨会话记忆（test14: store + namespace + put/search）
            └─> route_intent     结构化输出决策（test9: with_structured_output + 条件边）
                  ├─ small_talk     -> chat_direct      -> END
                  ├─ human_handoff  -> handoff          -> END
                  └─ rental_qa      -> generate_query_or_respond   （test2/test3: bind_tools）
                                          ├─ tools_condition = tools -> retrieve（ToolNode）
                                          │        └─> collect_sources
                                          │              └─> grade_documents（test3: 相关性评分，Command 路由）
                                          │                    ├─ yes -> generate_answer
                                          │                    └─ no  -> rewrite_question -> 回到检索
                                          └─ __end__ -> END
                                    generate_answer -> [human_review | summarize_conversation | END]
                                                        （test16-18 interrupt / test15 RemoveMessage）
"""
import uuid
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    filter_messages,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

import config
from agent import prompts
from agent.model import get_chat_model, structured
from agent.tools import TOOLS

# 触发历史消息摘要的消息条数上限
MAX_MESSAGES = int(getattr(config, "MAX_MESSAGES", 10) or 10)


# ============================================================ 状态
class RentalState(MessagesState):
    """在 MessagesState 之上扩展业务字段（LangGraph/test23.py 的写法）"""

    route: str                  # 路由决策
    context: str                # 检索到的上下文
    sources: list               # 来源明细，供 API 返回
    user_profile: str           # 跨会话记忆里读到的用户偏好
    rewrite_count: int          # 问题重写次数
    llm_calls: int              # 调用 LLM 的次数
    grade: str                  # 相关性评分结果（yes / no）


# ============================================================ 工具函数
def _last_human(messages) -> Optional[HumanMessage]:
    humans = filter_messages(messages, include_types="human")
    return humans[-1] if humans else None


def _user_id(config: RunnableConfig) -> str:
    if not config:
        return "anonymous"
    return config.get("configurable", {}).get("user_id", "anonymous")


# ============================================================ 节点 1：跨会话记忆
def sync_profile(state: RentalState, config: RunnableConfig, *, store: BaseStore):
    """
    从最新消息中提取租房偏好并写入 store；再把已有偏好读回状态。
    对应 LangGraph/test14.py：namespace = (user_id, "prefs")，put / search。
    """
    user_id = _user_id(config)
    namespace = (user_id, "prefs")

    human = _last_human(state["messages"])
    if human is not None:
        extractor = structured(prompts.UserProfile)
        profile = extractor.invoke([
            SystemMessage(content=prompts.PROFILE_SYSTEM),
            human,
        ])
        if any([profile.city, profile.budget, profile.room_type, profile.name]):
            store.put(namespace, str(uuid.uuid4()), profile.model_dump())

    memories = store.search(namespace, limit=3)
    merged: Dict[str, Any] = {}
    for m in memories:
        for k, v in (m.value or {}).items():
            if v:
                merged[k] = v

    profile_text = "；".join(f"{k}={v}" for k, v in merged.items()) or "(暂无历史偏好)"
    return {"user_profile": profile_text, "llm_calls": state.get("llm_calls", 0) + 1}


# ============================================================ 节点 2：意图路由
def route_intent(state: RentalState):
    """结构化输出决策（LangGraph/test9.py）"""
    human = _last_human(state["messages"])
    question = human.content if human else ""

    router = structured(prompts.Route)
    result = router.invoke([
        SystemMessage(content=prompts.ROUTER_SYSTEM),
        HumanMessage(content=question),
    ])
    return {"route": result.decision, "llm_calls": state.get("llm_calls", 0) + 1}


def route_decision(state: RentalState) -> str:
    mapping = {
        "rental_qa": "generate_query_or_respond",
        "small_talk": "chat_direct",
        "human_handoff": "handoff",
    }
    return mapping.get(state.get("route", "rental_qa"), "generate_query_or_respond")


# ============================================================ 节点 3：闲聊 / 转人工
def chat_direct(state: RentalState):
    result = get_chat_model().invoke(
        [SystemMessage(content=prompts.SMALL_TALK_SYSTEM)] + state["messages"]
    )
    return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}


def handoff(state: RentalState):
    result = get_chat_model().invoke(
        [SystemMessage(content=prompts.HANDOFF_SYSTEM)] + state["messages"]
    )
    return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}


# ============================================================ 节点 4：检索决策
def generate_query_or_respond(state: RentalState):
    """
    让模型决定：调用检索工具，还是直接回答。
    对应 LangGraph/test3.py 的 generate_query_or_respond + LangChain/test17.py 的 trim_messages。
    """
    messages = state["messages"]
    if len(messages) > MAX_MESSAGES:
        messages = trim_messages(
            messages,
            strategy="last",
            token_counter=len,
            max_tokens=MAX_MESSAGES,
            start_on="human",
            end_on=("human", "tool"),
        )

    model_with_tools = get_chat_model().bind_tools(TOOLS)
    result = model_with_tools.invoke(
        [SystemMessage(content="你可以使用 search_rental_knowledge 检索租房平台知识库。")]
        + messages
    )
    return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}


# ============================================================ 节点 5：收集来源
def collect_sources(state: RentalState):
    """
    ToolNode 只把工具输出塞进 ToolMessage；这里再取一次来源明细放进 state，
    方便 API 把引用出处返回给前端。
    """
    from ragbase import retrieval

    query = ""
    for msg in reversed(state["messages"]):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            query = tool_calls[0].get("args", {}).get("query", "")
            break

    context, sources = ("", [])
    if query:
        context, sources = retrieval.get_context(query, top_k=config.TOP_K)
    return {"context": context, "sources": sources}


# ============================================================ 节点 6：相关性评分
def grade_documents(state: RentalState) -> Command[Literal["generate_answer", "rewrite_question"]]:
    """
    评估检索到的文档是否与问题相关（LangGraph/test3.py 的相关性评分思路）。

    这里没有用 add_conditional_edges + 返回节点名的写法，而是让节点返回 Command：
    既做路由（goto），又能把评分结果和 LLM 调用次数写回 state。
    """
    # 已经重写过足够多次，不再死循环，直接生成
    if state.get("rewrite_count", 0) >= config.MAX_REWRITE:
        return Command(update={"grade": "yes"}, goto="generate_answer")

    human = _last_human(state["messages"])
    question = human.content if human else ""
    context = state.get("context", "")

    grader = structured(prompts.GradeDocuments)
    result = grader.invoke([
        HumanMessage(content=prompts.GRADE_PROMPT.format(context=context, question=question))
    ])

    score = "yes" if result.score == "yes" else "no"
    return Command(
        update={
            "grade": score,
            "llm_calls": state.get("llm_calls", 0) + 1,
        },
        goto="generate_answer" if score == "yes" else "rewrite_question",
    )


# ============================================================ 节点 7：问题重写
def rewrite_question(state: RentalState):
    """改写问题后重新检索（LangGraph/test3.py）"""
    human = _last_human(state["messages"])
    question = human.content if human else ""

    result = get_chat_model().invoke([
        HumanMessage(content=prompts.REWRITE_PROMPT.format(question=question))
    ])
    return {
        "messages": [HumanMessage(content=result.content)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================ 节点 8：生成答案
def generate_answer(state: RentalState):
    """基于上下文 + 少样本提示词生成最终答案（ChatPromptTemplate + FewShot）"""
    human = _last_human(state["messages"])
    question = human.content if human else ""

    chain = prompts.answer_prompt | get_chat_model()
    result = chain.invoke({
        "context": state.get("context", ""),
        "user_profile": state.get("user_profile", "(暂无历史偏好)"),
        "question": question,
    })
    return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}


# ============================================================ 节点 9：历史摘要
def summarize_conversation(state: RentalState):
    """消息过多时生成摘要并删除历史消息（LangGraph/test15.py）"""
    result = get_chat_model().invoke(
        state["messages"] + [HumanMessage(content="请用三句话以内总结上面的对话要点。")]
    )
    remove = [RemoveMessage(id=m.id) for m in state["messages"][:-1] if getattr(m, "id", None)]
    return {
        "messages": remove + [SystemMessage(content=f"历史对话摘要：{result.content}")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def after_answer(state: RentalState) -> str:
    if config.HUMAN_REVIEW:
        return "human_review"
    if len(state["messages"]) > MAX_MESSAGES:
        return "summarize_conversation"
    return END


# ============================================================ 节点 10：人工审核
def human_review(state: RentalState):
    """生成后的人工审核：可批准，也可直接编辑内容（LangGraph/test18.py）"""
    last = state["messages"][-1]
    decision = interrupt({
        "instruction": "请审核客服回复，可批准或改写",
        "content": getattr(last, "content", ""),
    })

    # decision 形如 {"approved": "true"/"false", "content": "..."}
    approved = str(decision.get("approved", "true")).lower() == "true"
    edited = (decision.get("content") or "").strip()

    if approved and not edited:
        return Command(goto=END)

    updates = []
    if getattr(last, "id", None):
        updates.append(RemoveMessage(id=last.id))
    updates.append(AIMessage(content=edited or getattr(last, "content", "")))
    return Command(update={"messages": updates}, goto=END)


# ============================================================ 构建图
def build_graph():
    builder = StateGraph(RentalState)

    builder.add_node(sync_profile)
    builder.add_node(route_intent)
    builder.add_node(chat_direct)
    builder.add_node(handoff)
    builder.add_node(generate_query_or_respond)
    builder.add_node("retrieve", ToolNode(TOOLS))
    builder.add_node(collect_sources)
    builder.add_node(grade_documents)
    builder.add_node(rewrite_question)
    builder.add_node(generate_answer)
    builder.add_node(summarize_conversation)
    builder.add_node(human_review)

    builder.add_edge(START, "sync_profile")
    builder.add_edge("sync_profile", "route_intent")

    builder.add_conditional_edges(
        "route_intent",
        route_decision,
        ["generate_query_or_respond", "chat_direct", "handoff"],
    )
    builder.add_edge("chat_direct", END)
    builder.add_edge("handoff", END)

    builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", "__end__": END},
    )
    builder.add_edge("retrieve", "collect_sources")

    # grade_documents 内部用 Command 做路由，所以这里是普通边
    builder.add_edge("collect_sources", "grade_documents")
    builder.add_edge("rewrite_question", "generate_query_or_respond")

    builder.add_conditional_edges(
        "generate_answer",
        after_answer,
        [END, "human_review", "summarize_conversation"],
    )
    builder.add_edge("summarize_conversation", END)

    return builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


graph = build_graph()


# ============================================================ 对外入口
def new_config(thread_id: str = "default", user_id: str = "anonymous") -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


def snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 graph.invoke 的原始结果整理成对外的返回结构。

    注意：state 里没被写过的 channel 读出来是 None（不是缺席），
    所以这里统一用 `or` 兜底，避免 API/CLI 返回 null。
    """
    messages = result.get("messages") or []
    return {
        "answer": messages[-1].content if messages else "",
        "sources": result.get("sources") or [],
        "route": result.get("route") or "",
        "rewrite_count": result.get("rewrite_count") or 0,
        "llm_calls": result.get("llm_calls") or 0,
        "user_profile": result.get("user_profile") or "",
        "grade": result.get("grade") or "",
        "interrupted": bool(result.get("__interrupt__")),
    }


def run(
    question: str,
    thread_id: str = "default",
    user_id: str = "anonymous",
) -> Dict[str, Any]:
    """同步执行一次问答，返回答案 + 来源 + 元信息"""
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=new_config(thread_id, user_id),
    )
    return snapshot(result)
